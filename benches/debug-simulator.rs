use nalgebra::{Complex, DMatrix};
use quasim::{debug_simulator::DebugSimulator, ext::equal_to_matrix_c};

extern crate quasim;

fn main() {
    divan::main();
}

#[divan::bench(
    args = [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    sample_count = 1,
    sample_size = 1,
)]
fn controlled_id_all_control_but_one(n_qubits: usize) {
    let n_controls = n_qubits - 1;

    let id_2x2 = DMatrix::<Complex<f32>>::identity(2, 2);
    let mat = DebugSimulator::expand_matrix(
        id_2x2,
        &(0..n_controls).collect::<Vec<usize>>(),
        &[n_controls],
        n_qubits,
    );
    let dim = 1 << n_qubits;
    let id_big = DMatrix::<Complex<f32>>::identity(dim, dim);

    assert!(equal_to_matrix_c(&mat, &id_big, 0.000001,));
}
