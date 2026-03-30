use cubecl::prelude::*;
use cubecl::{Runtime, bytes::Bytes, client::ComputeClient, server::Handle};
use nalgebra::Complex;
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::gate::QBits;
use crate::gate_batcher::BatchCommand;
use crate::gpu_sv_simulator::gpu_kernels;
use crate::{batched_circuit::BatchedCircuit, gpu_sv_simulator::mem_helpers};

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

    sampling_handle: Handle,
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
        let state_vector_len = 1 << n_qubits;
        let data_len = batched_circuit.data().len();

        let client = R::client(&Default::default());

        let gate_data = batched_circuit.data().gate_data();
        let target_data = batched_circuit.data().target_data();
        let control_data = batched_circuit.data().control_data();

        let gate_data_handle = client.create_from_slice(mem_helpers::bytes_from_complex(gate_data));
        let target_data_handle = client.create_from_slice(u32::as_bytes(target_data));
        let control_data_handle = client.create_from_slice(u32::as_bytes(control_data));

        // Temporary mem handle for sampling
        // Allocate enough for either probs (128 f32) or amplitudes (256 f32)
        let sampling_handle = client.empty(GPU_REDUCE_FACTOR * size_of::<f32>() * 2);

        // Reduction init
        let n_reduce_passes = n_qubits / GPU_REDUCE_FACTOR_EXP;
        let mut reduce_levels = Vec::new();
        let mut reduce_pass_size = state_vector_len / GPU_REDUCE_FACTOR;
        for _ in 0..n_reduce_passes {
            let handle = client.empty(reduce_pass_size * size_of::<f32>());
            reduce_levels.push(ReduceLevel {
                handle,
                len: reduce_pass_size,
            });

            reduce_pass_size /= GPU_REDUCE_FACTOR;
        }

        // State vector init
        let mut init_state: Vec<f32> = vec![0.0; state_vector_len * 2]; // two floats for each complex number
        init_state[0] = 1.0; // Sets first complex re = 1.0
        let init_bytes = Bytes::from_elems(init_state);
        let state_vector_handle = client.create_from_slice(&init_bytes);

        // Probs init
        let mut init_probs: Vec<f32> = vec![0.0; state_vector_len];
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
            sampling_handle,
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

    pub fn as_slice(&self) -> &[Complex<f32>] {
        unsafe { mem_helpers::complex_from_bytes(&self.state_vector_cache) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [Complex<f32>] {
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
                ArrayArg::from_raw_parts::<f32>(&self.state_vector_handle, n_amplitudes * 2, 1),
                ArrayArg::from_raw_parts::<u32>(&self.target_data_handle, self.data_len, 1),
                ArrayArg::from_raw_parts::<u32>(&self.control_data_handle, self.data_len, 1),
                ArrayArg::from_raw_parts::<f32>(&self.gate_data_handle, self.data_len * 8, 1),
                ScalarArg::new(command.start_index),
                ScalarArg::new(command.size),
                ScalarArg::new(command.targets.get_bitstring() as u32),
            );
        }
    }

    pub fn sample(&self) -> usize {
        self.build_reduction_hierarchy();

        let mut sample = 0;
        for (prob_handle, prob_len) in self
            .reduced_probs
            .iter()
            .rev()
            .map(|level| (&level.handle, level.len))
            .chain(std::iter::once((&self.probs_handle, self.state_vector_len)))
        {
            let local_index = self.sample_prob_block(prob_handle, prob_len, sample);
            sample = sample * GPU_REDUCE_FACTOR + local_index;
        }

        sample
    }

    pub fn measure_bits(&mut self, targets: QBits) -> usize {
        let measurement = self.sample() & targets.get_bitstring();

        let (cube_dim, cube_count) = self.cube_opts(self.state_vector_len);

        unsafe {
            let _ = gpu_kernels::state_vector_observe::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(
                    &self.state_vector_handle,
                    self.state_vector_len * 2,
                    1,
                ),
                ScalarArg::new(measurement as u32),
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
                ArrayArg::from_raw_parts::<f32>(
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

    fn build_reduction_hierarchy(&self) {
        self.launch_calculate_probs();

        let mut in_handle = &self.probs_handle;
        let mut in_len = self.state_vector_len;

        for out_level in &self.reduced_probs {
            self.launch_reduce_pass(in_len, in_handle, &out_level.handle);
            in_handle = &out_level.handle;
            in_len = out_level.len;
        }
    }

    fn sample_prob_block(&self, prob_handle: &Handle, prob_len: usize, sample: usize) -> usize {
        self.launch_copy_sampling_probs(prob_handle, prob_len, sample);

        let offset = sample * GPU_REDUCE_FACTOR;
        let valid_len = (prob_len - offset).min(GPU_REDUCE_FACTOR);
        let bytes = self.client.read_one(self.sampling_handle.clone());
        let weights = &f32::from_bytes(&bytes)[..valid_len];

        Self::sample_weights(weights.iter().copied())
    }

    fn sample_weights<I>(weights: I) -> usize
    where
        I: IntoIterator<Item = f32>,
    {
        let dist = WeightedIndex::new(weights)
            .expect("Failed to sample from state vector. Invalid or empty probability slice?");
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }

    fn normalize(&mut self) {
        // If state vector is smaller than `GPU_REDUCE_FACTOR`, then no reduce handles
        // will have been generated, and we can just do the normalization on the cpu
        if self.reduced_probs.is_empty() {
            self.sync_state_to_cpu();

            let norm = self
                .as_slice()
                .iter()
                .map(|x| x.norm_sqr())
                .sum::<f32>()
                .sqrt();

            self.as_slice_mut().iter_mut().for_each(|x| *x /= norm);

            self.sync_state_to_gpu();
        }
        // Else we do the normalization with reduce passes on the gpu:
        else {
            self.build_reduction_hierarchy();

            let final_level = self.reduced_probs.last().unwrap(); // safe unwrap
            let bytes = self.client.read_one(final_level.handle.clone());
            let reduced_sums = f32::from_bytes(&bytes);
            let norm = reduced_sums.iter().sum::<f32>().sqrt();

            self.launch_state_vector_division(norm);
        }
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
                ArrayArg::from_raw_parts::<f32>(&self.state_vector_handle, num_elems * 2, 1),
                ArrayArg::from_raw_parts::<f32>(&self.probs_handle, num_elems, 1),
            );
        }
    }

    fn launch_reduce_pass(&self, in_len: usize, in_handle: &Handle, out_handle: &Handle) {
        let cube_dim = CubeDim::new_1d(min(in_len, GPU_REDUCE_FACTOR) as u32);
        let cube_count = cubecl::calculate_cube_count_elemwise(&self.client, in_len, cube_dim);

        let out_len = in_len / GPU_REDUCE_FACTOR;

        unsafe {
            let _ = gpu_kernels::reduce_pass::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(in_handle, in_len, 1),
                ArrayArg::from_raw_parts::<f32>(out_handle, out_len, 1),
                GPU_REDUCE_FACTOR,
            );
        }
    }

    fn launch_state_vector_division(&self, norm: f32) {
        let num_elems = self.state_vector_len * 2;
        let (cube_dim, cube_count) = self.cube_opts(num_elems);

        unsafe {
            let _ = gpu_kernels::vector_division::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&self.state_vector_handle, num_elems, 1),
                ScalarArg::new(norm),
            );
        }
    }

    fn launch_copy_sampling_probs(&self, from_handle: &Handle, from_len: usize, sample: usize) {
        let num_elems = GPU_REDUCE_FACTOR;
        let (cube_dim, cube_count) = self.cube_opts(num_elems);

        let sample_offset = sample * num_elems;

        unsafe {
            let _ = gpu_kernels::copy_slice::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&from_handle, from_len, 1),
                ArrayArg::from_raw_parts::<f32>(&self.sampling_handle, num_elems, 1),
                ScalarArg::new(sample_offset),
            );
        }
    }
}
