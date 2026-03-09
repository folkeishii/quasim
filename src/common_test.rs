use nalgebra::dvector;

use crate::{
    cart,
    circuit::Circuit,
    ext::equal_to_matrix_c,
    simulator::{BuildSimulator, DebuggableSimulator, StoredCircuitSimulator},
};

#[allow(dead_code)]
#[allow(unreachable_code)]
// Multi control not gates not possible rn.
pub fn almost_grovers<D: BuildSimulator + DebuggableSimulator + StoredCircuitSimulator>() {
    // Keep for sub circuits
    return;
    const N: usize = 2;
    let sub = Circuit::new(N)
        // Step 1
        .hadamard(0)
        .z(1)
        // Step 2
        .cnot(0, 1)
        // Step 3
        .hadamard(0)
        .z(1);

    let mut circuit = Circuit::new(2 * N)
        // .new_sub_circuit("sub", sub)
        // Init
        .hadamard(0)
        .hadamard(1)
        .hadamard(2)
        .hadamard(3);

    const L: usize = 5; //(std::f64::consts::PI * 2f64.sqrt() / 4f64).floor() as usize;

    for _ in 0..L {
        // circuit = circuit.call("sub", 0);
        // circuit = circuit.call("sub", 2);
    }

    let mut sim = D::build(circuit).expect("Could not build simulator");

    assert!(equal_to_matrix_c(
        sim.cont(),
        &dvector![
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
            cart!(0.25),
        ],
        0.001
    ));
}
