use cubecl::cube;
use cubecl::prelude::*;

use crate::gpu_sv_simulator::GPU_MAX_BLOCK_SIZE;

#[derive(CubeType, Clone, Copy)]
struct ComplexF32 {
    re: f32,
    im: f32,
}

#[cube]
impl ComplexF32 {
    fn mul(self, rhs: Self) -> Self {
        ComplexF32 {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    fn add(self, rhs: Self) -> Self {
        ComplexF32 {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

/// Compute partial sums from an array with interleaved complex numbers
///
/// `block_size` should match number of units launched and is the number of complex amplitudes.
///
/// Assumes `state_vector`, `partial_sums`, and `block_size` is a power of 2.
#[cube(launch)]
pub fn reduce_complex(
    state_vector: &Array<f32>,
    partial_sums: &mut Array<f32>,
    #[comptime] block_size: usize,
) {
    let tid = UNIT_POS as usize;
    let bid = CUBE_POS;
    let gid = ABSOLUTE_POS;

    // Shared scratch memory between all threads in a block
    let mut shared: SharedMemory<f32> = SharedMemory::new(block_size);

    let amp_count = state_vector.len() / 2;

    // Initiate shared memory with the squared amplitudes
    shared[tid] = if gid < amp_count {
        let re = state_vector[2 * gid];
        let im = state_vector[2 * gid + 1];
        re * re + im * im
    } else {
        0.0.into()
    };

    sync_storage();

    let mut stride = block_size >> 1;
    while stride > 0 {
        if tid < stride {
            shared[tid] += shared[tid + stride];
        }

        stride >>= 1;
        sync_storage();
    }

    if tid == 0 {
        partial_sums[bid] = shared[0];
    }
}

#[cube(launch)]
pub fn reduce_pass(
    partial_in: &Array<f32>,
    partial_out: &mut Array<f32>,
    #[comptime] block_size: usize,
) {
    let tid = UNIT_POS as usize;
    let bid = CUBE_POS;
    let gid = ABSOLUTE_POS;

    // Shared scratch memory between all threads in a block
    let mut shared: SharedMemory<f32> = SharedMemory::new(block_size);

    let partial_len = partial_in.len();

    shared[tid] = if gid < partial_len {
        partial_in[gid]
    } else {
        0.0.into()
    };

    sync_storage();

    let mut stride = block_size >> 1;
    while stride > 0 {
        if tid < stride {
            shared[tid] += shared[tid + stride];
        }

        stride >>= 1;
        sync_storage();
    }

    if tid == 0 {
        partial_out[bid] = shared[0];
    }
}

/// Applies `size` number of gates in one batch
#[cube(launch)]
pub fn batched_gate2(
    state_vector: &mut Array<f32>,
    target_data: &Array<u32>,
    control_data: &Array<u32>,
    gate_data: &Array<f32>,
    start_index: u32,
    size: u32,
    target_mask: u32,
) {
    let block_size: usize = 1usize << (target_mask.count_ones() as usize);
    let state_vector_len: usize = state_vector.len() / 2;

    // One kernel is launcher for each block, so
    // num units = statevector length / block size
    let num_units: usize = state_vector_len / block_size;
    if ABSOLUTE_POS < num_units {
        let mut block_indices: Array<usize> = Array::new(GPU_MAX_BLOCK_SIZE);
        for i in 0..block_size {
            block_indices[i] = block_index(ABSOLUTE_POS, target_mask as usize, i);
        }

        for i in 0..size {
            let data_index: usize = (start_index + i) as usize;
            let target: usize = target_data[data_index] as usize;
            let control: usize = control_data[data_index] as usize;
            let gate_base = data_index * 8;

            let u00 = ComplexF32 {
                re: gate_data[gate_base],
                im: gate_data[gate_base + 1],
            };
            let u01 = ComplexF32 {
                re: gate_data[gate_base + 2],
                im: gate_data[gate_base + 3],
            };
            let u10 = ComplexF32 {
                re: gate_data[gate_base + 4],
                im: gate_data[gate_base + 5],
            };
            let u11 = ComplexF32 {
                re: gate_data[gate_base + 6],
                im: gate_data[gate_base + 7],
            };

            for local_index in 0..block_size {
                let block_index = block_indices[local_index];

                apply_unitary2(
                    state_vector,
                    block_index,
                    target,
                    control,
                    u00,
                    u01,
                    u10,
                    u11,
                );
            }
        }
    }
}

/// Applies a gate to specified basis state on state vector
#[cube]
fn apply_unitary2(
    state_vector: &mut Array<f32>,
    basis_state: usize,
    target: usize,
    control: usize,
    u00: ComplexF32,
    u01: ComplexF32,
    u10: ComplexF32,
    u11: ComplexF32,
) {
    let basis0: usize = basis_state * 2;
    let basis1: usize = (basis_state | target) * 2;

    let is_block_base: bool = (basis_state & target) == 0;
    let controls_active: bool = (basis_state & control) == control;

    if is_block_base && controls_active {
        let amp0 = ComplexF32 {
            re: state_vector[basis0],
            im: state_vector[basis0 + 1],
        };
        let amp1 = ComplexF32 {
            re: state_vector[basis1],
            im: state_vector[basis1 + 1],
        };

        let new_amp0 = u00.mul(amp0).add(u01.mul(amp1));
        let new_amp1 = u10.mul(amp0).add(u11.mul(amp1));

        state_vector[basis0] = new_amp0.re;
        state_vector[basis0 + 1] = new_amp0.im;

        state_vector[basis1] = new_amp1.re;
        state_vector[basis1 + 1] = new_amp1.im;
    }
}

/// Returns the global block index in statevector for a specific superblock,
/// given target mask and local index in the superblock.
#[cube]
fn block_index(mut superblock: usize, mut target_mask: usize, local_index: usize) -> usize {
    let mut shift = 0usize;
    while target_mask != 0 {
        // AND target_mask with wrapping neg of itself to get lowest bit
        let lowest = target_mask & ((!target_mask) + 1);
        // Get position of lowest bit by counting ones equivalent of leading zeroes
        let pos = (lowest - 1).count_ones() as usize;
        // Take next bit from local_index
        let bit = (local_index >> shift) & 1;
        let lower = superblock & ((1 << pos) - 1);
        let upper = superblock >> pos;
        // Reconstruct blockindex
        superblock = lower | (bit << pos) | (upper << (pos + 1));
        target_mask &= target_mask - 1;
        shift += 1;
    }
    superblock
}
