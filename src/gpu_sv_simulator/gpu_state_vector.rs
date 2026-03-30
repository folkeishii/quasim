use cubecl::prelude::*;
use cubecl::{Runtime, bytes::Bytes, client::ComputeClient, server::Handle};
use nalgebra::Complex;

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

    reduce_levels: Vec<ReduceLevel>,
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
            reduce_levels,
        }
    }

    pub fn sync_state_to_cpu(&mut self) {
        self.state_vector_cache = self.client.read_one(self.state_vector_handle.clone());
    }

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

        let cube_dim = CubeDim::new_1d(min(n_superblocks, GPU_MAX_CUBE_DIM) as u32);
        let cube_count =
            cubecl::calculate_cube_count_elemwise(&self.client, n_superblocks, cube_dim);

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

    fn build_reduction_tree(&self) {
        for (i, out_level) in self.reduce_levels.iter().enumerate() {
            // First iteration we always have to reduce with complex numbers in mind
            if i == 0 {
                self.launch_reduce_state_vector(out_level);
            } else {
                let in_level = &self.reduce_levels[i - 1];
                self.launch_reduce_pass(in_level, out_level);
            }
        }
    }

    fn normalize(&mut self) {
        // If state vector is smaller than `GPU_REDUCE_FACTOR`, then no reduce handles
        // will have been generated, and we can just do the normalization on the cpu
        if self.reduce_levels.is_empty() {
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
            self.build_reduction_tree();

            let final_level = self.reduce_levels.last().unwrap(); // safe unwrap
            let bytes = self.client.read_one(final_level.handle.clone());
            let reduced_sums = f32::from_bytes(&bytes);
            let norm = reduced_sums.iter().sum::<f32>().sqrt();

            self.launch_state_vector_divide(norm);
        }
    }

    // Kernel launch wrappers

    fn launch_reduce_state_vector(&self, out_level: &ReduceLevel) {
        let num_elems = self.state_vector_len;
        let cube_dim = CubeDim::new_1d(min(num_elems, GPU_REDUCE_FACTOR) as u32);
        let cube_count = cubecl::calculate_cube_count_elemwise(&self.client, num_elems, cube_dim);

        unsafe {
            let _ = gpu_kernels::reduce_complex::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&self.state_vector_handle, num_elems * 2, 1),
                ArrayArg::from_raw_parts::<f32>(&out_level.handle, out_level.len, 1),
                GPU_REDUCE_FACTOR,
            );
        }
    }

    fn launch_reduce_pass(&self, in_level: &ReduceLevel, out_level: &ReduceLevel) {
        let num_elems = in_level.len;
        let cube_dim = CubeDim::new_1d(min(num_elems, GPU_REDUCE_FACTOR) as u32);
        let cube_count = cubecl::calculate_cube_count_elemwise(&self.client, num_elems, cube_dim);

        unsafe {
            let _ = gpu_kernels::reduce_pass::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&in_level.handle, in_level.len, 1),
                ArrayArg::from_raw_parts::<f32>(&out_level.handle, out_level.len, 1),
                GPU_REDUCE_FACTOR,
            );
        }
    }

    fn launch_state_vector_divide(&self, norm: f32) {
        let num_elems = self.state_vector_len * 2;
        let cube_dim = CubeDim::new_1d(min(num_elems, GPU_MAX_CUBE_DIM) as u32);
        let cube_count = cubecl::calculate_cube_count_elemwise(&self.client, num_elems, cube_dim);

        unsafe {
            let _ = gpu_kernels::array_divide::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&self.state_vector_handle, num_elems, 1),
                ScalarArg::new(norm),
            );
        }
    }

    fn launch_copy_sampling_probs(&self, level: ReduceLevel, sample: usize) {
        let num_elems = GPU_REDUCE_FACTOR;
        let cube_dim = CubeDim::new_1d(min(num_elems, GPU_MAX_CUBE_DIM) as u32);
        let cube_count = cubecl::calculate_cube_count_elemwise(&self.client, num_elems, cube_dim);

        let sample_offset = sample * num_elems;

        unsafe {
            let _ = gpu_kernels::copy_offset::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&level.handle, level.len, 1),
                ArrayArg::from_raw_parts::<f32>(&self.sampling_handle, num_elems, 1),
                ScalarArg::new(sample_offset),
            );
        }
    }


    fn launch_copy_sampling_amplitudes(&self, sample: usize) {
        let num_elems = GPU_REDUCE_FACTOR * 2;
        let cube_dim = CubeDim::new_1d(min(num_elems, GPU_MAX_CUBE_DIM) as u32);
        let cube_count = cubecl::calculate_cube_count_elemwise(&self.client, num_elems, cube_dim);

        let sample_offset = sample * num_elems;

        unsafe {
            let _ = gpu_kernels::copy_offset::launch(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&self.state_vector_handle, self.state_vector_len * 2, 1),
                ArrayArg::from_raw_parts::<f32>(&self.sampling_handle, num_elems, 1),
                ScalarArg::new(sample_offset),
            );
        }
    }
}
