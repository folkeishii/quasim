use std::f32::consts::{FRAC_1_SQRT_2, PI};

use nalgebra::{Complex, DVector, dvector};
use rand::random;

use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit},
    expr_dsl::expr_helpers::r,
    ext::{equal_state_c, expand_matrix_from_gate},
    gate::{Gate, GateType},
    sampler::{CircuitSampler, RegisterSampler},
    simulator::{Buildable, Debuggable, QuantumState, Sampleable, StoredRegisters},
    state_vector::StateVector,
};

pub fn apply_gates<Sim: Buildable<PureCircuit> + Debuggable>()
where
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
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

    let mut sim = Sim::build(circuit).expect("Could not build simulator");

    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.state()}, &StateVector::from(s1), 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.state()}, &StateVector::from(s2), 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.state()}, &StateVector::from(s3), 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.state()}, &StateVector::from(s4), 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.state()}, &StateVector::from(s5), 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.state()}, &StateVector::from(s6), 3, 0.000001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.state()}, &StateVector::from(s7), 3, 0.000001));

    apply_cswap::<Sim>();
}

fn apply_cswap<Sim: Buildable<PureCircuit> + Debuggable>()
where
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
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
    let mut sim = Sim::build(circuit).expect("Could not build simulator");
    sim.next();
    sim.next();
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.state()}, &StateVector::from(s1), 4, 0.00001));
    #[rustfmt::skip]
    assert!(equal_state_c({sim.next();sim.state()}, &StateVector::from(s2), 4, 0.00001));
}

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

    circuit = circuit.z(2 * N - 1);

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
            cart!(-0.25),
            cart!(-0.25),
            cart!(-0.25),
            cart!(-0.25),
            cart!(-0.25),
            cart!(-0.25),
            cart!(-0.25),
            cart!(-0.25),
        ]),
        2 * N,
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

fn state_vector_from_quantum_state<Sim>(state: &Sim::State, n_qubits: usize) -> StateVector
where
    Sim: crate::simulator::Simulator,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    let mut actual = StateVector::zeros(n_qubits);
    for basis in 0..(1 << n_qubits) {
        actual[basis] = state.basis_value(basis).into();
    }
    actual
}

pub fn hadamard_cnot_entanglement_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    let circuit = Circuit::new(2).h(0).cx(&[0], 1);
    for _ in 0..1000 {
        let mut sim = match Sim::build(circuit.clone()) {
            Ok(sim) => sim,
            Err(error) => panic!("Error building simulator: {}", error),
        };
        sim.run();
        let result = sim.state().collapse();
        assert!(result == 0b00 || result == 0b11);
    }
}

pub fn probability_distribution_sums_to_one_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    let mut circuit = Circuit::new(2).h(0).cx(&[0], 1);

    for i in 0..10 {
        circuit = circuit.h(0).u(random(), random(), random(), i % 2);
    }

    let mut sim = match Sim::build(circuit) {
        Ok(sim) => sim,
        Err(error) => panic!("Error building simulator: {}", error),
    };
    sim.run();
    let total_probability: f32 = (0..4)
        .map(|basis| sim.state().basis_value(basis).norm_sqr())
        .sum();
    assert!(
        (total_probability - 1.0).abs() < 1e-4,
        "Total probability does not sum to 1, got {}",
        total_probability
    );
}

pub fn ry_inversion_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    let circuit = Circuit::new(1).ry(PI, 0);
    for _ in 0..1000 {
        let mut sim = match Sim::build(circuit.clone()) {
            Ok(sim) => sim,
            Err(error) => panic!("Error building simulator: {}", error),
        };
        sim.run();
        let result = sim.state().collapse();
        assert_eq!(result, 1);
    }
}

pub fn invert_qubit_through_rz_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    let circuit = Circuit::new(1).h(0).rz(PI, 0).h(0);
    for _ in 0..1000 {
        let mut sim = match Sim::build(circuit.clone()) {
            Ok(sim) => sim,
            Err(error) => panic!("Error building simulator: {}", error),
        };
        sim.run();
        let result = sim.state().collapse();
        assert_eq!(result, 1);
    }
}

pub fn swap_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    const N: usize = 10;
    for n_qubits in 1..=N {
        let mut circuit = Circuit::new(n_qubits).h(0).z(0);
        for i in 0..(n_qubits - 1) {
            circuit = circuit.swap(i, i + 1);
        }

        circuit = circuit.h(n_qubits - 1);

        let mut sim = match Sim::build(circuit.clone()) {
            Ok(sim) => sim,
            Err(error) => panic!("Error building simulator: {}", error),
        };
        sim.run();
        let result = sim.state().collapse();
        assert_eq!(result, 1 << (n_qubits - 1));
    }
}

/// The QFT turns the most significant bit into |+⟩ if the parity of the input is even,
/// and into |−⟩ if the parity is odd, so measuring it after hadamard will give 0 for
/// even parity and 1 for odd parity.
pub fn parity_by_qft_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    let n = 5;
    let qft_targets = (0..n).collect::<Vec<_>>();
    let circuit = Circuit::new(n);
    for i in 0..2usize.pow(n as u32) {
        let mut circuit_with_input = circuit.clone();
        let mut j = 0;
        while (1 << j) <= i {
            if i & (1 << j) != 0 {
                circuit_with_input = circuit_with_input.x(j);
            }
            j += 1;
        }
        let mut sim = match Sim::build(circuit_with_input.qft(&qft_targets).h(n - 1)) {
            Ok(sim) => sim,
            Err(error) => panic!("Error building simulator: {}", error),
        };
        sim.run();
        let result = sim.state().collapse();
        assert_eq!(
            result >> (n - 1),
            i % 2,
            "Failed on input {} which resulted in {}",
            i,
            result
        );
    }
}

pub fn swap_respects_controls_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    let circuit = Circuit::new(3).x(1).h(0).cswap(&[0], 1, 2);
    for _ in 0..1000 {
        let mut sim = match Sim::build(circuit.clone()) {
            Ok(sim) => sim,
            Err(error) => panic!("Error building simulator: {}", error),
        };
        sim.run();
        let result = sim.state().collapse();
        assert!(
            result == 0b010 || result == 0b101,
            "Failed with result {:03b}",
            result
        );
    }
}

pub fn respects_global_phase_against_matrix_multiplication_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    let circuit = Circuit::new(1).x(0).s(0).h(0);

    let mut simulator = match Sim::build(circuit.clone()) {
        Ok(simulator) => simulator,
        Err(error) => panic!("Error building simulator: {}", error),
    };
    simulator.run();

    let mut expected = StateVector::zeros(1);
    for gate in [
        Gate::new(GateType::X, &[], &[0]).unwrap(),
        Gate::new(GateType::S, &[], &[0]).unwrap(),
        Gate::new(GateType::H, &[], &[0]).unwrap(),
    ] {
        expected.apply_matrix(&expand_matrix_from_gate(&gate, 1));
    }

    let actual = state_vector_from_quantum_state::<Sim>(simulator.state(), 1);

    assert!(equal_state_c(&actual, &expected, 1, 0.001));
}

pub fn matches_matrix_multiplication_for_single_qubit_gate_types_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    let circuit = Circuit::new(1)
        .x(0)
        .y(0)
        .z(0)
        .h(0)
        .s(0)
        .u(PI / 3.0, PI / 4.0, -PI / 5.0, 0);

    let mut simulator = match Sim::build(circuit.clone()) {
        Ok(simulator) => simulator,
        Err(error) => panic!("Error building simulator: {}", error),
    };
    simulator.run();

    let mut expected = StateVector::zeros(1);
    for gate in [
        Gate::new(GateType::X, &[], &[0]).unwrap(),
        Gate::new(GateType::Y, &[], &[0]).unwrap(),
        Gate::new(GateType::Z, &[], &[0]).unwrap(),
        Gate::new(GateType::H, &[], &[0]).unwrap(),
        Gate::new(GateType::S, &[], &[0]).unwrap(),
        Gate::new(GateType::U(PI / 3.0, PI / 4.0, -PI / 5.0), &[], &[0]).unwrap(),
    ] {
        expected.apply_matrix(&expand_matrix_from_gate(&gate, 1));
    }

    let actual = state_vector_from_quantum_state::<Sim>(simulator.state(), 1);

    assert!(equal_state_c(&actual, &expected, 1, 0.001));
}

pub fn matches_matrix_multiplication_for_swap_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    let circuit = Circuit::new(2).x(0).h(1).swap(0, 1);

    let mut simulator = match Sim::build(circuit.clone()) {
        Ok(simulator) => simulator,
        Err(error) => panic!("Error building simulator: {}", error),
    };
    simulator.run();

    let mut expected = StateVector::zeros(2);
    for gate in [
        Gate::new(GateType::X, &[], &[0]).unwrap(),
        Gate::new(GateType::H, &[], &[1]).unwrap(),
        Gate::new(GateType::SWAP, &[], &[0, 1]).unwrap(),
    ] {
        expected.apply_matrix(&expand_matrix_from_gate(&gate, 2));
    }

    let actual = state_vector_from_quantum_state::<Sim>(simulator.state(), 2);

    assert!(equal_state_c(&actual, &expected, 2, 0.001));
}

fn next_random_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn random_index(state: &mut u64, upper_bound: usize) -> usize {
    next_random_u64(state) as usize % upper_bound
}

fn random_angle(state: &mut u64) -> f32 {
    let unit = next_random_u64(state) as f32 / u64::MAX as f32;
    (unit * 2.0 - 1.0) * PI
}

pub fn random_circuits_match_matrix_multiplication_pure<Sim>()
where
    Sim: Buildable<PureCircuit>,
    Sim::State: QuantumState<BasisValue = Complex<f32>>,
{
    const N_QUBITS: usize = 3;
    const N_TRIALS: usize = 16;
    const N_STEPS: usize = 12;

    let seed = random::<u64>();
    let mut rng_state = seed;

    for trial in 0..N_TRIALS {
        let mut circuit = Circuit::new(N_QUBITS).x(0).x(1).x(2);
        let mut expected = StateVector::zeros(N_QUBITS);
        let mut sequence = vec![
            String::from("x(0)"),
            String::from("x(1)"),
            String::from("x(2)"),
        ];
        for gate in [
            Gate::new(GateType::X, &[], &[0]).unwrap(),
            Gate::new(GateType::X, &[], &[1]).unwrap(),
            Gate::new(GateType::X, &[], &[2]).unwrap(),
        ] {
            expected.apply_matrix(&expand_matrix_from_gate(&gate, N_QUBITS));
        }

        for _ in 0..N_STEPS {
            let choice = random_index(&mut rng_state, 14);
            let target = random_index(&mut rng_state, N_QUBITS);

            match choice {
                0 => {
                    circuit = circuit.x(target);
                    sequence.push(format!("x({})", target));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::X, &[], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                1 => {
                    circuit = circuit.y(target);
                    sequence.push(format!("y({})", target));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::Y, &[], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                2 => {
                    circuit = circuit.z(target);
                    sequence.push(format!("z({})", target));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::Z, &[], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                3 => {
                    circuit = circuit.h(target);
                    sequence.push(format!("h({})", target));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::H, &[], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                4 => {
                    circuit = circuit.s(target);
                    sequence.push(format!("s({})", target));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::S, &[], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                5 => {
                    let theta = random_angle(&mut rng_state);
                    let phi = random_angle(&mut rng_state);
                    let lambda = random_angle(&mut rng_state);
                    circuit = circuit.u(theta, phi, lambda, target);
                    sequence.push(format!(
                        "u({:.3}, {:.3}, {:.3}, {})",
                        theta, phi, lambda, target
                    ));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::U(theta, phi, lambda), &[], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                6 => {
                    let control = (target + 1) % N_QUBITS;
                    circuit = circuit.cx(&[control], target);
                    sequence.push(format!("cx([{}], {})", control, target));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::X, &[control], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                7 => {
                    let control = (target + 1) % N_QUBITS;
                    circuit = circuit.cy(&[control], target);
                    sequence.push(format!("cy([{}], {})", control, target));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::Y, &[control], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                8 => {
                    let control = (target + 1) % N_QUBITS;
                    circuit = circuit.cz(&[control], target);
                    sequence.push(format!("cz([{}], {})", control, target));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::Z, &[control], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                9 => {
                    let control = (target + 1) % N_QUBITS;
                    circuit = circuit.ch(&[control], target);
                    sequence.push(format!("ch([{}], {})", control, target));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::H, &[control], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                10 => {
                    let control = (target + 1) % N_QUBITS;
                    circuit = circuit.cs(&[control], target);
                    sequence.push(format!("cs([{}], {})", control, target));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::S, &[control], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                11 => {
                    let control = (target + 1) % N_QUBITS;
                    let theta = random_angle(&mut rng_state);
                    let phi = random_angle(&mut rng_state);
                    let lambda = random_angle(&mut rng_state);
                    circuit = circuit.cu(theta, phi, lambda, &[control], target);
                    sequence.push(format!(
                        "cu({:.3}, {:.3}, {:.3}, [{}], {})",
                        theta, phi, lambda, control, target
                    ));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::U(theta, phi, lambda), &[control], &[target]).unwrap(),
                        N_QUBITS,
                    ));
                }
                12 => {
                    let other = (target + 1) % N_QUBITS;
                    circuit = circuit.swap(target, other);
                    sequence.push(format!("swap({}, {})", target, other));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::SWAP, &[], &[target, other]).unwrap(),
                        N_QUBITS,
                    ));
                }
                _ => {
                    let control = (target + 1) % N_QUBITS;
                    let t1 = control;
                    let t2 = (target + 2) % N_QUBITS;
                    circuit = circuit.cswap(&[target], t1, t2);
                    sequence.push(format!("cswap([{}], {}, {})", target, t1, t2));
                    expected.apply_matrix(&expand_matrix_from_gate(
                        &Gate::new(GateType::SWAP, &[target], &[t1, t2]).unwrap(),
                        N_QUBITS,
                    ));
                }
            }
        }

        let mut simulator = match Sim::build(circuit) {
            Ok(simulator) => simulator,
            Err(error) => panic!("Error building simulator: {}", error),
        };
        simulator.run();

        let actual = state_vector_from_quantum_state::<Sim>(simulator.state(), N_QUBITS);

        assert!(
            equal_state_c(&actual, &expected, N_QUBITS, 0.01),
            "Random circuit mismatch on trial {} with seed 0x{:016X}: {:?}",
            trial,
            seed,
            sequence
        );
    }
}
