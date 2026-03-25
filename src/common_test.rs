use std::f64::consts::FRAC_1_SQRT_2;

use nalgebra::{Complex, DVector, dvector};

use crate::{
    cart,
    circuit::{Circuit, HybridCircuit},
    ext::equal_to_matrix_c,
    simulator::{BuildSimulator, DebuggableSimulator, StoredCircuitSimulator},
};

pub fn double_sub<
    D: BuildSimulator<HybridCircuit> + DebuggableSimulator + StoredCircuitSimulator,
>() {
    // Keep for sub circuits
    const N: usize = 2;
    let sub = Circuit::new(N)
        // Step 1
        .h(0)
        .z(1)
        // Step 2
        .cx(&[0], 1)
        // Step 3
        .h(0)
        .z(1);

    let mut circuit = Circuit::new(2 * N)
        .new_sub_circuit("sub", sub)
        // Init
        .h(0)
        .h(1)
        .h(2)
        .h(3);

    const L: usize = 5; //(std::f64::consts::PI * 2f64.sqrt() / 4f64).floor() as usize;

    for _ in 0..L {
        circuit = circuit.call("sub", 0);
        circuit = circuit.call("sub", 2);
    }

    let mut sim = D::build(circuit.into()).expect("Could not build simulator");

    let mut forward_steps = 0;
    while sim.next().is_some() {
        forward_steps += 1;
    }

    assert!(equal_to_matrix_c(
        sim.current_state(),
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

    if sim.double_ended() {
        let mut backward_steps = 0;
        while sim.prev().is_some() {
            backward_steps += 1;
        }
        assert_eq!(forward_steps, backward_steps);
        assert!(equal_to_matrix_c(
            sim.current_state(),
            &dvector![
                cart!(1),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
                cart!(0),
            ],
            0.001
        ));
    }
}

pub fn deep_sub<D: BuildSimulator<HybridCircuit> + DebuggableSimulator + StoredCircuitSimulator>() {
    // Keep for sub circuits
    const LEVELS: usize = 5;
    let subs1 = Circuit::new(1).h(0);
    let subs2 = Circuit::new(2)
        .new_sub_circuit("sub 1", subs1)
        .call("sub 1", 1);
    let subs3 = Circuit::new(3)
        .new_sub_circuit("sub 2", subs2)
        .call("sub 2", 1);
    let subs4 = Circuit::new(4)
        .new_sub_circuit("sub 3", subs3)
        .call("sub 3", 1);
    let subs5 = Circuit::new(5)
        .new_sub_circuit("sub 4", subs4)
        .call("sub 4", 1);

    let circuit = Circuit::new(LEVELS)
        .new_sub_circuit("sub 5", subs5)
        .call("sub 5", 0);

    let mut sim = D::build(circuit.into()).expect("Could not build simulator");

    let mut forward_steps = 0;
    while sim.next().is_some() {
        forward_steps += 1;
    }

    let mut correct = DVector::<Complex<f64>>::zeros(1 << LEVELS);
    correct[0] = cart!(FRAC_1_SQRT_2);
    correct[1 << (LEVELS - 1)] = cart!(FRAC_1_SQRT_2);

    assert!(equal_to_matrix_c(sim.current_state(), &correct, 0.001));

    if sim.double_ended() {
        let mut backward_steps = 0;
        while sim.prev().is_some() {
            backward_steps += 1;
        }
        assert_eq!(forward_steps, backward_steps);

        let mut correct = DVector::<Complex<f64>>::zeros(1 << LEVELS);
        correct[0] = cart!(1);
        assert!(equal_to_matrix_c(sim.current_state(), &correct, 0.001));
    }
}
