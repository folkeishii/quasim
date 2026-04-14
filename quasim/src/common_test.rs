use std::{f64::consts::FRAC_1_SQRT_2};

use nalgebra::{Complex, DVector, dvector};

use crate::{
    cart,
    circuit::{Circuit, HybridCircuit},
    expr_dsl::expr_helpers::r,
    ext::equal_state_c,
    sampler::{CircuitSampler, RegisterSampler},
    simulator::{Buildable, Debuggable, QuantumState, Sampleable, StoredRegisters},
};

pub fn double_sub<Sim: Buildable<HybridCircuit> + Debuggable>()
where
    Sim::State: QuantumState<BasisValue = Complex<f64>>,
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

    let mut sim = Sim::build(circuit.into()).expect("Could not build simulator");

    let mut forward_steps = 0;
    while sim.next() {
        forward_steps += 1;
    }

    assert!(equal_state_c(
        sim.state(),
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
            sim.state(),
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

pub fn deep_sub<Sim: Buildable<HybridCircuit> + Debuggable>()
where
    Sim::State: QuantumState<BasisValue = Complex<f64>>,
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

    let mut sim = Sim::build(circuit.into()).expect("Could not build simulator");

    let mut forward_steps = 0;
    while sim.next() {
        forward_steps += 1;
    }

    let mut correct = DVector::<Complex<f64>>::zeros(1 << LEVELS);
    correct[0] = cart!(FRAC_1_SQRT_2);
    correct[1 << (LEVELS - 1)] = cart!(FRAC_1_SQRT_2);

    assert!(equal_state_c(sim.state(), &correct, LEVELS, 0.001));

    if sim.double_ended() {
        let mut backward_steps = 0;
        while sim.prev() {
            backward_steps += 1;
        }
        assert_eq!(forward_steps, backward_steps);

        let mut correct = DVector::<Complex<f64>>::zeros(1 << LEVELS);
        correct[0] = cart!(1);
        assert!(equal_state_c(sim.state(), &correct, LEVELS, 0.001));
    }
}

pub fn hybrid_test<Sim: Buildable<HybridCircuit>>()
where
    Sim::State: QuantumState<BasisValue = Complex<f64>>,
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

    let mut sim = Sim::build(circuit).unwrap();
    sim.run();

    let mut expected = DVector::<Complex<f64>>::zeros(16);
    expected[0] = cart!(1.0);

    assert!(equal_state_c(sim.state(), &expected, 4, 0.001));
}

pub fn register_test<Sim: Sampleable<HybridCircuit> + StoredRegisters>() {
    let circuit = Circuit::new(2)
        .new_reg("r0", 1)
        .x(1)
        .measure_bit(1, ("r0", 0));

    assert_eq!(
        Sim::sample_once(circuit, RegisterSampler::new("r0")).unwrap(),
        1
    );
}

pub fn test_measure_overwrites_with_zero<Sim>()
where
    Sim: Sampleable<HybridCircuit> + StoredRegisters,
{
    let circuit = Circuit::new(2)
        .new_reg("tmp", 1)
        .x(0)
        .measure_bit(0, ("tmp", 0))
        .measure_bit(1, ("tmp", 0));

    assert_eq!(
        Sim::sample_once(circuit, RegisterSampler::new("tmp")).unwrap(),
        0
    );
}

pub fn test_reset<Sim>()
where
    Sim: Buildable<HybridCircuit> + StoredRegisters,
{
    let circuit = Circuit::new(4)
        .new_reg("r0", 2)
        .new_reg("r1", 2)
        .x(0)
        .x(3)
        .measure_bits(&[0, 1], "r0")
        .measure_bits(&[2, 3], "r1");

    let mut sim = Sim::build(circuit).unwrap();

    sim.run();
    assert_eq!(sim.register("r0").read(), 0b01);
    assert_eq!(sim.register("r1").read(), 0b10);
    assert_eq!(sim.state().collapse(), 0b1001);
    sim.reset();
    assert_eq!(sim.register("r0").read(), 0);
    assert_eq!(sim.register("r1").read(), 0);
    assert_eq!(sim.state().collapse(), 0);
}

pub fn test_reset_with_shared_scratch_register<Sim>()
where
    Sim: Sampleable<HybridCircuit>,
{
    let circuit = Circuit::new(4)
        .h(0)
        .h(1)
        .h(2)
        .h(3)
        .reset(0)
        .reset(1)
        .reset(2)
        .reset(3);

    assert!(
        Sim::sample(circuit, CircuitSampler, 100)
            .unwrap()
            .fold(true, |acc, it| acc && it == 0)
    )
}

pub fn deep_ctrl_sub<Sim: Buildable<HybridCircuit> + Debuggable>()
where
    Sim::State: QuantumState<BasisValue = Complex<f64>>,
{
    // Keep for sub circuits
    const LEVELS: usize = 5;
    let subs1 = Circuit::new(1).x(0).h(0);
    let subs2 = Circuit::new(2).x(0).h(0).ccall_new("sub 1", subs1, 1, &[0]);
    let subs3 = Circuit::new(3).x(0).h(0).ccall_new("sub 2", subs2, 1, &[0]);
    let subs4 = Circuit::new(4).x(0).h(0).ccall_new("sub 3", subs3, 1, &[0]);

    let circuit = Circuit::new(LEVELS).h(0).ccall_new("sub 4", subs4, 1, &[0]);
    let mut sim = Sim::build(circuit.into()).expect("Could not build simulator");

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

    assert!(equal_state_c(sim.state(), &correct, 5, 0.001));

    if sim.double_ended() {
        let mut backward_steps = 0;
        while sim.prev() {
            backward_steps += 1;
        }
        assert_eq!(forward_steps, backward_steps);

        let mut correct = DVector::<Complex<f64>>::zeros(1 << LEVELS);
        correct[0] = cart!(1);
        assert!(equal_state_c(sim.state(), &correct, LEVELS, 0.001));
    }
}
