use cubecl::prelude::*;
use cubecl::{Runtime, bytes::Bytes, client::ComputeClient, server::Handle};
use nalgebra::Complex;
use rand::Rng;

use crate::gate::QBits;
use crate::gpu_sv_simulator::batched_circuit::BatchedCircuit;
use crate::gpu_sv_simulator::gate_batcher::BatchCommand;
use crate::gpu_sv_simulator::{gpu_kernels, mem_helpers};

const GPU_REDUCE_FACTOR_EXP: usize = 7;
const GPU_REDUCE_FACTOR: usize = 1 << GPU_REDUCE_FACTOR_EXP;
const GPU_MAX_CUBE_DIM: usize = 128;

#[derive(Clone)]
pub struct GpuStateVector<R: Runtime> {
    client: ComputeClient<R>,

    state_vector_cache: Bytes,
    state_vector_len: usize,
    state_vector_handle: Handle,

    data_len: usize,
    gate_data_handle: Handle,
    target_data_handle: Handle,
    control_data_handle: Handle,

    probs_handle: Handle,
    reduced_probs: Vec<ReduceLevel>,
}

#[derive(Clone)]
struct ReduceLevel {
    handle: Handle,
    len: usize,
}

impl<R: Runtime> GpuStateVector<R> {
    pub fn new(n_qubits: usize, batched_circuit: &BatchedCircuit) -> Self {
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
            data_len,
            gate_data_handle,
            target_data_handle,
            control_data_handle,
            probs_handle,
            reduced_probs: reduce_levels,
        }
    }

    /// Avoid using this function unless necessary, downloads whole state vector from gpu
    pub fn sync_state_to_cpu(&mut self) {
        self.state_vector_cache = self.client.read_one(self.state_vector_handle.clone());
    }

    /// Avoid using this function unless necessary, uploads whole state vector to gpu
    pub fn sync_state_to_gpu(&mut self) {
        self.state_vector_handle = self.client.create(self.state_vector_cache.clone());
    }

    pub fn as_slice(&self) -> &[Complex<f64>] {
        unsafe { mem_helpers::complex_from_bytes(&self.state_vector_cache) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [Complex<f64>] {
        unsafe { mem_helpers::complex_from_bytes_mut(&mut self.state_vector_cache) }
    }

    pub fn apply_batch_command(&self, command: &BatchCommand) {
        let n_amplitudes = self.state_vector_len;
        let batch_target_count = command.targets.count();

        let n_superblocks = n_amplitudes >> batch_target_count;

        let (cube_dim, cube_count) = self.cube_opts(n_superblocks);

        unsafe {
            let _ = gpu_kernels::batched_gate2::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f64>(&self.state_vector_handle, n_amplitudes * 2, 1),
                ArrayArg::from_raw_parts::<u32>(&self.target_data_handle, self.data_len, 1),
                ArrayArg::from_raw_parts::<u32>(&self.control_data_handle, self.data_len, 1),
                ArrayArg::from_raw_parts::<f64>(&self.gate_data_handle, self.data_len * 8, 1),
                ScalarArg::new(command.start_index),
                ScalarArg::new(command.size),
                ScalarArg::new(command.targets.get_bitstring() as u32),
            );
        }
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

        let bytes = self.client.read_one(sample_index_handle);
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
                ArrayArg::from_raw_parts::<f64>(
                    &self.state_vector_handle,
                    self.state_vector_len * 2,
                    1,
                ),
                ScalarArg::new(measurement as u32),
                ScalarArg::new(measurement_mask),
            );
        }
        self.normalize();

        measurement
    }

    pub fn measure(&mut self) -> usize {
        let measurement = self.sample();

        let (cube_dim, cube_count) = self.cube_opts(self.state_vector_len);

        unsafe {
            let _ = gpu_kernels::state_vector_observe_full::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f64>(
                    &self.state_vector_handle,
                    self.state_vector_len * 2,
                    1,
                ),
                ScalarArg::new(measurement as u32),
            );
        }

        measurement
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

    fn normalize(&self) {
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
                ArrayArg::from_raw_parts::<f64>(&self.state_vector_handle, num_elems * 2, 1),
                ArrayArg::from_raw_parts::<f64>(&self.probs_handle, num_elems, 1),
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
                ArrayArg::from_raw_parts::<f64>(in_handle, in_len, 1),
                ArrayArg::from_raw_parts::<f64>(out_handle, out_len, 1),
                GPU_REDUCE_FACTOR,
            );
        }
    }

    fn launch_state_vector_normalize(&self) {
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
                ArrayArg::from_raw_parts::<f64>(&self.state_vector_handle, num_elems, 1),
                ArrayArg::from_raw_parts::<f64>(&final_sum_level.handle, 1, 1),
            );
        }
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
                ArrayArg::from_raw_parts::<f64>(prob_handle, prob_len, 1),
                ArrayArg::from_raw_parts::<u32>(sample_index_handle, 1, 1),
                ArrayArg::from_raw_parts::<f64>(sample_threshold_handle, 1, 1),
                GPU_REDUCE_FACTOR,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use cubecl::{bytes::Bytes, wgpu::WgpuRuntime};

    use super::*;
    use crate::circuit::Circuit;

    fn make_state_vector(n_qubits: usize) -> GpuStateVector<WgpuRuntime> {
        GpuStateVector::new(n_qubits, &BatchedCircuit::from(Circuit::new(n_qubits)))
    }

    #[test]
    fn observe_respects_zero_bits_in_target_mask() {
        let mut state = make_state_vector(2);

        let init = Bytes::from_elems(vec![
            1.0f64, 0.0, // |00>
            1.0, 0.0, // |01>
            0.0, 0.0, // |10>
            0.0, 0.0, // |11>
        ]);
        state.state_vector_cache = init.clone();
        state.state_vector_handle = state.client.create(init);

        let (cube_dim, cube_count) = state.cube_opts(state.state_vector_len);

        unsafe {
            let _ = gpu_kernels::state_vector_observe::launch(
                &state.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f64>(
                    &state.state_vector_handle,
                    state.state_vector_len * 2,
                    1,
                ),
                ScalarArg::new(0u32),
                ScalarArg::new(1u32),
            );
        }

        state.sync_state_to_cpu();
        let amps = state.as_slice();

        assert_eq!(amps[0], Complex::new(1.0, 0.0));
        assert_eq!(amps[1], Complex::new(0.0, 0.0));
        assert_eq!(amps[2], Complex::new(0.0, 0.0));
        assert_eq!(amps[3], Complex::new(0.0, 0.0));
    }

    #[test]
    fn observe_full_collapses_to_exact_basis_state() {
        let mut state = make_state_vector(2);

        let init = Bytes::from_elems(vec![
            0.0f64, 0.0, // |00>
            0.0, 0.0, // |01>
            1.0, 0.0, // |10>
            0.0, 0.0, // |11>
        ]);
        state.state_vector_cache = init.clone();
        state.state_vector_handle = state.client.create(init);

        let (cube_dim, cube_count) = state.cube_opts(state.state_vector_len);

        unsafe {
            let _ = gpu_kernels::state_vector_observe_full::launch(
                &state.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f64>(
                    &state.state_vector_handle,
                    state.state_vector_len * 2,
                    1,
                ),
                ScalarArg::new(2u32),
            );
        }

        state.sync_state_to_cpu();
        let amps = state.as_slice();

        assert_eq!(amps[0], Complex::new(0.0, 0.0));
        assert_eq!(amps[1], Complex::new(0.0, 0.0));
        assert_eq!(amps[2], Complex::new(1.0, 0.0));
        assert_eq!(amps[3], Complex::new(0.0, 0.0));
    }
}
