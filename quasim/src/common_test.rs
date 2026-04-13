use std::{f64::consts::FRAC_1_SQRT_2, ops::Index};

use nalgebra::{Complex, DVector, dvector};

use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit},
    expr_dsl::expr_helpers::r,
    ext::equal_state_c,
    simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator, StoredCircuitSimulator},
};

pub fn apply_gates<D>()
where
    D: BuildSimulator<PureCircuit> + DebuggableSimulator,
    D::Storage: Index<usize, Output = Complex<f64>>,
{
    let mut circuit = Circuit::new(3);
    circuit = circuit.h(0);
    let s1 = dvector![
        cart!(FRAC_1_SQRT_2),
        cart!(FRAC_1_SQRT_2),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0)
    ];
    circuit = circuit.y(1);
    let s2 = dvector![
        cart!(0),
        cart!(0),
        cart!(0, FRAC_1_SQRT_2),
        cart!(0, FRAC_1_SQRT_2),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0)
    ];
    circuit = circuit.z(2);
    let s3 = s2.clone();
    circuit = circuit.z(1);
    let s4 = dvector![
        cart!(0),
        cart!(0),
        cart!(0, -FRAC_1_SQRT_2),
        cart!(0, -FRAC_1_SQRT_2),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0)
    ];
    circuit = circuit.ch(&[0], 1);
    let s5 = dvector![
        cart!(0),
        cart!(0, -0.5),
        cart!(0, -FRAC_1_SQRT_2),
        cart!(0, 0.5),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0)
    ];
    circuit = circuit.cy(&[0, 1], 2);
    let s6 = dvector![
        cart!(0),
        cart!(0, -0.5),
        cart!(0, -FRAC_1_SQRT_2),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(-0.5),
    ];
    circuit = circuit.swap(0, 2);
    let s7 = dvector![
        cart!(0),
        cart!(0),
        cart!(0, -FRAC_1_SQRT_2),
        cart!(0),
        cart!(0, -0.5),
        cart!(0),
        cart!(0),
        cart!(-0.5),
    ];

    let mut sim = D::build(circuit).expect("Could not build simulator");

    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.current_state()}, &s1, 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.current_state()}, &s2, 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.current_state()}, &s3, 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.current_state()}, &s4, 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.current_state()}, &s5, 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.current_state()}, &s6, 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.current_state()}, &s7, 3, 0.000001));

    apply_cswap::<D>();
}

fn apply_cswap<D>()
where
    D: BuildSimulator<PureCircuit> + DebuggableSimulator,
    D::Storage: Index<usize, Output = Complex<f64>>,
{
    let mut circuit = Circuit::new(4);
    circuit = circuit.h(0).h(1).h(2);
    let s1 = dvector![
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
    ];
    circuit = circuit.cswap(&[0, 2], 1, 3);
    let s2 = dvector![
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0.35355),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0),
        cart!(0.35355),
        cart!(0),
        cart!(0),
    ];
    let mut sim = D::build(circuit).expect("Could not build simulator");
    sim.next();
    sim.next();
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.current_state()}, &s1, 4, 0.00001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.current_state()}, &s2, 4, 0.00001));
}

pub fn double_sub<D: BuildSimulator<HybridCircuit> + DebuggableSimulator + StoredCircuitSimulator>()
where
    D::Storage: Index<usize, Output = Complex<f64>>,
{
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
    while sim.next() {
        forward_steps += 1;
    }

    assert!(equal_state_c(
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
        N,
        0.001
    ));

    if sim.double_ended() {
        let mut backward_steps = 0;
        while sim.prev() {
            backward_steps += 1;
        }
        assert_eq!(forward_steps, backward_steps);
        assert!(equal_state_c(
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
            N,
            0.001
        ));
    }
}

pub fn deep_sub<D: BuildSimulator<HybridCircuit> + DebuggableSimulator + StoredCircuitSimulator>()
where
    D::Storage: Index<usize, Output = Complex<f64>>,
{
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
    while sim.next() {
        forward_steps += 1;
    }

    let mut correct = DVector::<Complex<f64>>::zeros(1 << LEVELS);
    correct[0] = cart!(FRAC_1_SQRT_2);
    correct[1 << (LEVELS - 1)] = cart!(FRAC_1_SQRT_2);

    assert!(equal_state_c(sim.current_state(), &correct, LEVELS, 0.001));

    if sim.double_ended() {
        let mut backward_steps = 0;
        while sim.prev() {
            backward_steps += 1;
        }
        assert_eq!(forward_steps, backward_steps);

        let mut correct = DVector::<Complex<f64>>::zeros(1 << LEVELS);
        correct[0] = cart!(1);
        assert!(equal_state_c(sim.current_state(), &correct, LEVELS, 0.001));
    }
}

pub fn hybrid_test<
    D: BuildSimulator<HybridCircuit> + DebuggableSimulator + StoredCircuitSimulator,
>()
where
    D::Storage: Index<usize, Output = Complex<f64>>,
{
    let circuit = Circuit::new(4)
        .new_reg("r0", 1)
        .new_reg("r1", 1)
        .new_reg("r2", 1)
        .new_reg("r3", 1)
        // Init random state
        .h(0)
        .h(1)
        .h(2)
        .h(3)
        .measure_bit(0, ("r0", 0))
        .measure_bit(1, ("r1", 0))
        .measure_bit(2, ("r2", 0))
        .measure_bit(3, ("r3", 0))
        .apply_if(r("r0").eq(1))
        .x(0)
        .apply_if(r("r1").eq(1))
        .x(1)
        .apply_if(r("r2").eq(1))
        .x(2)
        .apply_if(r("r3").eq(1))
        .x(3);

    let mut sim = D::build(circuit).unwrap();
    while sim.next() {}

    let mut expected = DVector::<Complex<f64>>::zeros(16);
    expected[0] = cart!(1.0);

    assert!(equal_state_c(sim.current_state(), &expected, 4, 0.001));
}

pub fn register_test<
    D: BuildSimulator<HybridCircuit> + DebuggableSimulator + StoredCircuitSimulator + HybridSimulator,
>() {
    let circuit = Circuit::new(2)
        .new_reg("r0", 1)
        .x(1)
        .measure_bit(1, ("r0", 0));

    let mut sim = D::build(circuit).unwrap();
    while sim.next() {}

    assert_eq!(sim.registers()["r0"].read(), 1);
}

pub fn test_measure_overwrites_with_zero<T>()
where
    T: BuildSimulator<HybridCircuit>
        + DebuggableSimulator
        + StoredCircuitSimulator
        + HybridSimulator,
{
    let circuit = Circuit::new(2)
        .new_reg("tmp", 1)
        .x(0)
        .measure_bit(0, ("tmp", 0))
        .measure_bit(1, ("tmp", 0));

    let mut sim = T::build(circuit).unwrap();
    sim.cont();

    assert_eq!(sim.register("tmp").read(), 0);
}

pub fn deep_ctrl_sub<
    D: BuildSimulator<HybridCircuit> + DebuggableSimulator + StoredCircuitSimulator,
>()
where
    D::Storage: Index<usize, Output = Complex<f64>>,
{
    // Keep for sub circuits
    const LEVELS: usize = 5;
    let subs1 = Circuit::new(1).x(0).h(0);
    let subs2 = Circuit::new(2).x(0).h(0).ccall_new("sub 1", subs1, 1, &[0]);
    let subs3 = Circuit::new(3).x(0).h(0).ccall_new("sub 2", subs2, 1, &[0]);
    let subs4 = Circuit::new(4).x(0).h(0).ccall_new("sub 3", subs3, 1, &[0]);

    let circuit = Circuit::new(LEVELS).h(0).ccall_new("sub 4", subs4, 1, &[0]);
    let mut sim = D::build(circuit.into()).expect("Could not build simulator");

    let mut forward_steps = 0;
    while sim.next() {
        forward_steps += 1;
    }

    let mut correct = DVector::<Complex<f64>>::zeros(1 << LEVELS);
    correct[0b00000] = cart!(FRAC_1_SQRT_2);
    correct[0b00001] = cart!(0.5);
    correct[0b00011] = cart!(-0.35355);
    correct[0b00111] = cart!(0.25);
    correct[0b01111] = cart!(-0.17678);
    correct[0b11111] = cart!(0.17678);

    assert!(equal_state_c(sim.current_state(), &correct, 5, 0.001));

    if sim.double_ended() {
        let mut backward_steps = 0;
        while sim.prev() {
            backward_steps += 1;
        }
        assert_eq!(forward_steps, backward_steps);

        let mut correct = DVector::<Complex<f64>>::zeros(1 << LEVELS);
        correct[0] = cart!(1);
        assert!(equal_state_c(sim.current_state(), &correct, LEVELS, 0.001));
    }
}
