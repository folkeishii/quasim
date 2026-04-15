use std::f32::consts::FRAC_1_SQRT_2;

use nalgebra::{Complex, DVector, dvector};

use crate::{
    cart,
    circuit::{Circuit, HybridCircuit},
    expr_dsl::expr_helpers::r,
    ext::equal_state_c,
    sampler::{CircuitSampler, RegisterSampler},
    simulator::{Buildable, Debuggable, QuantumState, Sampleable, StoredRegisters},
    state_vector::StateVector,
};

pub fn double_sub<Sim: Buildable<HybridCircuit> + Debuggable>()
where
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    println!("double sub");
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

    const L: usize = 5; //(std::f32::consts::PI * 2f32.sqrt() / 4f32).floor() as usize;

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
        &StateVector::from(dvector![
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
        ]),
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
            &StateVector::from(dvector![
                cart!(1.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
            ]),
            N,
            0.001
        ));
    }
}

pub fn deep_sub<Sim: Buildable<HybridCircuit> + Debuggable>()
where
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    println!("deep sub");
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

    let mut correct: StateVector = DVector::<Complex<f32>>::zeros(1 << LEVELS).into();
    correct[0] = cart!(FRAC_1_SQRT_2);
    correct[1 << (LEVELS - 1)] = cart!(FRAC_1_SQRT_2);
    println!("deep sub ok");

    assert!(equal_state_c(sim.state(), &correct, LEVELS, 0.001));

    if sim.double_ended() {
        let mut backward_steps = 0;
        while sim.prev() {
            backward_steps += 1;
        }
        assert_eq!(forward_steps, backward_steps);

        let correct = StateVector::zeros(LEVELS);
        assert!(equal_state_c(sim.state(), &correct, LEVELS, 0.001));
    }
}

pub fn hybrid_test<Sim: Buildable<HybridCircuit>>()
where
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    println!("hybrid test");
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

    let expected = StateVector::zeros(16);
    assert!(equal_state_c(sim.state(), &expected, 4, 0.001));
}

pub fn register_test<Sim: Sampleable<HybridCircuit> + StoredRegisters>() {
    println!("register test");
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
    println!("test measure overwrites with zero");
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
    println!("test reset");
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
    println!("test reset with shared scratch register");
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
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    println!("deep ctrl sub");
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

    let mut correct: StateVector = DVector::<Complex<f32>>::zeros(1 << LEVELS).into();

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

        let correct = StateVector::zeros(LEVELS);

        assert!(equal_state_c(sim.state(), &correct, LEVELS, 0.001));
    }
}
pub fn mid_measure_all<Sim: Sampleable<HybridCircuit> + StoredRegisters>() {
    println!("mid measure all");
    let circuit = Circuit::new(5)
        .new_reg("a", 4)
        .new_reg("~a", 4)
        .h(0)
        .h(1)
        .h(2)
        .h(3)
        .measure("a")
        .x(0)
        .x(1)
        .x(2)
        .x(3)
        .measure("~a")
        .apply_if((r("a") + r("~a")).eq(0b1111))
        .x(4)
        .new_reg("res", 1)
        .measure_bit(4, ("res", 0));
    assert!(Sim::sample_once(circuit, RegisterSampler::new("res")).unwrap() & 1 == 1);
}

pub fn mid_measure_bit<Sim: Sampleable<HybridCircuit> + StoredRegisters>() {
    println!("mid measure bit");
    let circuit = Circuit::new(5)
        .new_reg("a", 4)
        .new_reg("~a", 4)
        .h(0)
        .h(1)
        .h(2)
        .h(3)
        .measure_bit(0, ("a", 0))
        .measure_bit(1, ("a", 1))
        .measure_bit(2, ("a", 2))
        .measure_bit(3, ("a", 3))
        .x(0)
        .x(1)
        .x(2)
        .x(3)
        .measure_bit(0, ("~a", 0))
        .measure_bit(1, ("~a", 1))
        .measure_bit(2, ("~a", 2))
        .measure_bit(3, ("~a", 3))
        .apply_if((r("a") + r("~a")).eq(0b1111))
        .x(4)
        .new_reg("res", 1)
        .measure_bit(4, ("res", 0));
    assert!(Sim::sample_once(circuit, RegisterSampler::new("res")).unwrap() & 1 == 1);
}

pub fn interleaved<Sim: Buildable<HybridCircuit> + StoredRegisters>()
where
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    println!("interleaved");
    let mut sim = Sim::build(
        Circuit::new(4)
            .h(0)
            .ch(&[0], 2)
            .swap(0, 2)
            .h(1)
            .ch(&[1], 3)
            .swap(0, 3)
            .ch(&[2], 1)
            .swap(2, 3)
            .ch(&[0], 3)
            .swap(0, 1)
            .into(),
    )
    .unwrap();
    sim.run();

    let expected: StateVector = dvector![
        cart!(0.5000000293365844),
        cart!(0.35355340368276855),
        cart!(0.12500000042912138),
        cart!(0.12500000042912138),
        cart!(0.0),
        cart!(0.0),
        cart!(0.12500000042912138),
        cart!(-0.12500000042912138),
        cart!(0.4267766656533139),
        cart!(0.07322330885223931),
        cart!(-0.12500000042912138),
        cart!(0.37500001339455813),
        cart!(0.4267766656533139),
        cart!(0.07322330885223931),
        cart!(-0.12500000042912138),
        cart!(0.12500000042912138),
    ]
    .into();

    assert!(equal_state_c(sim.state(), &expected, 4, 0.001));
}
