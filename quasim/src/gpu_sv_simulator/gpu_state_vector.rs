use std::fmt::Display;
use std::ops::Index;

use cubecl::prelude::*;
use cubecl::{Runtime, bytes::Bytes, client::ComputeClient, server::Handle};
use nalgebra::{Complex, DVectorView};
use rand::Rng;

use crate::gate::QBits;
use crate::gpu_sv_simulator::batched_circuit::BatchedCircuit;
use crate::gpu_sv_simulator::gate_batcher::BatchCommand;
use crate::gpu_sv_simulator::{gpu_kernels, mem_helpers};
use crate::simulator::QuantumState;

const GPU_REDUCE_FACTOR_EXP: usize = 7;
const GPU_REDUCE_FACTOR: usize = 1 << GPU_REDUCE_FACTOR_EXP;
const GPU_MAX_CUBE_DIM: usize = 128;

#[derive(Clone)]
pub struct GpuStateVector<R: Runtime> {
    client: ComputeClient<R>,

    state_vector_cache: Bytes,
    state_vector_len: usize,
    state_vector_handle: Handle,
    state_vector_dirty: bool,

    data_len: usize,
    gate_data_handle: Handle,
    target_data_handle: Handle,
    control_data_handle: Handle,

    probs_handle: Handle,
    reduced_probs: Vec<ReduceLevel>,

    max_block_size: usize,
}

#[derive(Clone)]
struct ReduceLevel {
    handle: Handle,
    len: usize,
}

impl<R: Runtime> QuantumState for GpuStateVector<R> {
    type BasisValue = Complex<f64>;

    fn collapse(&self) -> usize {
        self.sample()
    }

    /// Requires state vector to be synced to cpu in order to get updated values
    fn basis_value(&self, basis: usize) -> Self::BasisValue {
        self.as_slice()[basis]
    }
}

impl<R: Runtime> Index<usize> for GpuStateVector<R> {
    type Output = Complex<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<R: Runtime> Display for GpuStateVector<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dvec_view = DVectorView::from_slice(self.as_slice(), self.state_vector_len);

        write!(f, "{}", dvec_view)?;

        if self.state_vector_dirty {
            write!(f, "\n(dirty)")?;
        }
        Ok(())
    }
}

impl<R: Runtime> GpuStateVector<R> {
    pub fn new<const MAX_QUBITS_PER_BATCH: usize>(
        batched_circuit: &BatchedCircuit<MAX_QUBITS_PER_BATCH>,
    ) -> Self {
        let n_qubits = batched_circuit.n_qubits();
        let state_vector_len: usize = 1 << n_qubits;
        let data_len = batched_circuit.data().len();

        let client = R::client(&Default::default());

        let gate_data = batched_circuit.data().gate_data();
        let target_data = batched_circuit.data().target_data();
        let control_data = batched_circuit.data().control_data();

        let gate_data_handle = client.create_from_slice(mem_helpers::bytes_from_complex(gate_data));
        let target_data_handle = client.create_from_slice(u32::as_bytes(target_data));
        let control_data_handle = client.create_from_slice(u32::as_bytes(control_data));

        // Reduction init
        // Last reduction level will always be a single sized handle with the complete sum
        let n_reduce_passes = n_qubits.div_ceil(GPU_REDUCE_FACTOR_EXP);
        let mut reduce_levels = Vec::new();
        let mut reduce_pass_size = state_vector_len.div_ceil(GPU_REDUCE_FACTOR);
        for _ in 0..n_reduce_passes {
            let handle = client.empty(reduce_pass_size * size_of::<f64>());
            reduce_levels.push(ReduceLevel {
                handle,
                len: reduce_pass_size,
            });

            reduce_pass_size = reduce_pass_size.div_ceil(GPU_REDUCE_FACTOR);
        }

        // State vector init
        let mut init_state: Vec<f64> = vec![0.0; state_vector_len * 2]; // two floats for each complex number
        init_state[0] = 1.0; // Sets first complex re = 1.0
        let init_bytes = Bytes::from_elems(init_state);
        let state_vector_handle = client.create_from_slice(&init_bytes);

        // Probs init
        let mut init_probs: Vec<f64> = vec![0.0; state_vector_len];
        init_probs[0] = 1.0; // Sets first state prob = 1.0
        let init_probs = Bytes::from_elems(init_probs);
        let probs_handle = client.create_from_slice(&init_probs);

        Self {
            client,
            state_vector_cache: init_bytes,
            state_vector_len,
            state_vector_handle,
            state_vector_dirty: false,
            data_len,
            gate_data_handle,
            target_data_handle,
            control_data_handle,
            probs_handle,
            reduced_probs: reduce_levels,
            max_block_size: 1 << MAX_QUBITS_PER_BATCH,
        }
    }

    /// Downloads whole state vector from gpu
    pub fn sync_state_to_cpu(&mut self) {
        if self.state_vector_dirty {
            self.state_vector_cache = self
                .client
                .read_one(self.state_vector_handle.clone())
                .expect("failed to read state back to CPU");
        }

        self.state_vector_dirty = false;
    }

    /// Uploads whole state vector to gpu
    pub fn sync_state_to_gpu(&mut self) {
        self.state_vector_handle = self.client.create(self.state_vector_cache.clone());
    }

    pub fn as_slice(&self) -> &[Complex<f64>] {
        unsafe { mem_helpers::complex_from_bytes(&self.state_vector_cache) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [Complex<f64>] {
        unsafe { mem_helpers::complex_from_bytes_mut(&mut self.state_vector_cache) }
    }

    pub fn apply_batch_command(&mut self, command: &BatchCommand) {
        let n_amplitudes = self.state_vector_len;
        let batch_target_count = command.targets.count();

        let n_superblocks = n_amplitudes >> batch_target_count;

        let (cube_dim, cube_count) = self.cube_opts(n_superblocks);

        unsafe {
            let _ = gpu_kernels::batched_gate2::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(self.state_vector_handle.clone(), n_amplitudes * 2),
                ArrayArg::from_raw_parts(self.target_data_handle.clone(), self.data_len),
                ArrayArg::from_raw_parts(self.control_data_handle.clone(), self.data_len),
                ArrayArg::from_raw_parts(self.gate_data_handle.clone(), self.data_len * 8),
                command.start_index,
                command.size,
                command.targets.get_bitstring() as u32,
                self.max_block_size,
            );
        }

        self.state_vector_dirty = true;
    }

    pub fn sample(&self) -> usize {
        self.launch_calculate_probs();
        self.build_prob_reduction_hierarchy();

        let threshold = rand::rng().random_range(0.0..1.0f64);
        let sample_index_handle = self.client.create_from_slice(u32::as_bytes(&[0]));
        let sample_threshold_handle = self.client.create_from_slice(f64::as_bytes(&[threshold]));

        for (prob_handle, prob_len) in self
            .reduced_probs
            .iter()
            .rev()
            .map(|level| (&level.handle, level.len))
            .chain(std::iter::once((&self.probs_handle, self.state_vector_len)))
        {
            self.launch_sample_cdf_block(
                prob_handle,
                prob_len,
                &sample_index_handle,
                &sample_threshold_handle,
            );
        }

        let bytes = self
            .client
            .read_one(sample_index_handle)
            .expect("failed to read sample result back to CPU");
        u32::from_bytes(&bytes)[0] as usize
    }

    pub fn measure_bits(&mut self, targets: QBits) -> usize {
        let measurement = self.sample() & targets.get_bitstring();
        let measurement_mask = targets.get_bitstring() as u32;

        let (cube_dim, cube_count) = self.cube_opts(self.state_vector_len);

        unsafe {
            let _ = gpu_kernels::state_vector_observe::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(
                    self.state_vector_handle.clone(),
                    self.state_vector_len * 2,
                ),
                measurement as u32,
                measurement_mask,
            );
        }
        self.normalize();
        self.state_vector_dirty = true;

        measurement
    }

    pub fn measure_all(&mut self) -> usize {
        let measurement = self.sample();

        self.launch_state_vector_observe_full(measurement);

        measurement
    }

    pub fn reset(&mut self) {
        self.launch_state_vector_observe_full(0);
    }

    // Helpers

    fn cube_opts(&self, num_elems: usize) -> (CubeDim, CubeCount) {
        let cube_dim = CubeDim::new_1d(min(num_elems, GPU_MAX_CUBE_DIM) as u32);
        let cube_count = cubecl::calculate_cube_count_elemwise(&self.client, num_elems, cube_dim);

        (cube_dim, cube_count)
    }

    fn build_prob_reduction_hierarchy(&self) {
        let mut in_handle = &self.probs_handle;
        let mut in_len = self.state_vector_len;

        for out_level in &self.reduced_probs {
            self.launch_reduce_pass(in_len, in_handle, &out_level.handle);
            in_handle = &out_level.handle;
            in_len = out_level.len;
        }
    }

    fn normalize(&mut self) {
        self.launch_calculate_probs();
        self.build_prob_reduction_hierarchy();
        self.launch_state_vector_normalize();
    }

    // Kernel launch wrappers

    fn launch_calculate_probs(&self) {
        let num_elems = self.state_vector_len;
        let (cube_dim, cube_count) = self.cube_opts(num_elems);

        unsafe {
            let _ = gpu_kernels::calculate_probs::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(self.state_vector_handle.clone(), num_elems * 2),
                ArrayArg::from_raw_parts(self.probs_handle.clone(), num_elems),
            );
        }
    }

    fn launch_reduce_pass(&self, in_len: usize, in_handle: &Handle, out_handle: &Handle) {
        let cube_dim = CubeDim::new_1d(GPU_REDUCE_FACTOR as u32);
        let cube_count = cubecl::calculate_cube_count_elemwise(&self.client, in_len, cube_dim);

        let out_len = in_len.div_ceil(GPU_REDUCE_FACTOR);

        unsafe {
            let _ = gpu_kernels::reduce_pass::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(in_handle.clone(), in_len),
                ArrayArg::from_raw_parts(out_handle.clone(), out_len),
                GPU_REDUCE_FACTOR,
            );
        }
    }

    fn launch_state_vector_normalize(&mut self) {
        let num_elems = self.state_vector_len * 2;
        let (cube_dim, cube_count) = self.cube_opts(num_elems);

        let final_sum_level = self.reduced_probs.last().unwrap();

        // the last level should only contain one float, the total sum
        assert_eq!(final_sum_level.len, 1);

        unsafe {
            let _ = gpu_kernels::state_vector_normalize::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(self.state_vector_handle.clone(), num_elems),
                ArrayArg::from_raw_parts(final_sum_level.handle.clone(), 1),
            );
        }
        self.state_vector_dirty = true;
    }

    /// Expects probs to already be normalized
    fn launch_sample_cdf_block(
        &self,
        prob_handle: &Handle,
        prob_len: usize,
        sample_index_handle: &Handle,
        sample_threshold_handle: &Handle,
    ) {
        let cube_dim = CubeDim::new_1d(GPU_REDUCE_FACTOR as u32);
        let cube_count = CubeCount::Static(1, 1, 1);

        unsafe {
            let _ = gpu_kernels::sample_cdf_block::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(prob_handle.clone(), prob_len),
                ArrayArg::from_raw_parts(sample_index_handle.clone(), 1),
                ArrayArg::from_raw_parts(sample_threshold_handle.clone(), 1),
                GPU_REDUCE_FACTOR,
            );
        }
    }

    fn launch_state_vector_observe_full(&mut self, measurement: usize) {
        let (cube_dim, cube_count) = self.cube_opts(self.state_vector_len);

        unsafe {
            let _ = gpu_kernels::state_vector_observe_full::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(
                    self.state_vector_handle.clone(),
                    self.state_vector_len * 2,
                ),
                measurement as u32,
            );
        }
        self.state_vector_dirty = true;
    }
}
