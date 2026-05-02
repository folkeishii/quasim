use std::f32::consts::PI;

use rand::random;

use crate::{
    circuit::Circuit,
    ext::{equal_state_c, expand_matrix_from_gate},
    gate::QBits,
    gate::{Gate, GateType},
    simulator::{Buildable, QuantumState},
    state_vector::StateVector,
    syntax_simulator::{
        basis::{ExtendedBasis, ExtendedQubitBasis},
        scalar::Scalar,
    },
};

use super::*;

#[test]
fn test_expand_qubits() {
    let basis_to_expand =
        ExtendedBasis::Superposition(vec![ExtendedQubitBasis::Plus, ExtendedQubitBasis::Minus]);

    use ExtendedQubitBasis::*;
    assert_eq!(
        basis_to_expand.clone().expand_qubit(0),
        [
            Some(ScaledState(
                ExtendedBasis::Superposition(vec![Zero, Minus]),
                Scalar::FRAC_1_SQRT_2
            )),
            Some(ScaledState(
                ExtendedBasis::Superposition(vec![One, Minus]),
                Scalar::FRAC_1_SQRT_2
            )),
        ]
    );

    assert_eq!(
        basis_to_expand.clone().expand_qubit(1),
        [
            Some(ScaledState(
                ExtendedBasis::Superposition(vec![Plus, Zero]),
                Scalar::FRAC_1_SQRT_2
            )),
            Some(ScaledState(
                ExtendedBasis::Superposition(vec![Plus, One]),
                -Scalar::FRAC_1_SQRT_2
            )),
        ]
    );

    let expand_both = QBits::from_indices(&[0, 1]);
    let half = Scalar::FRAC_1_SQRT_2 * Scalar::FRAC_1_SQRT_2;

    let correct = vec![
        ScaledState(ExtendedBasis::Superposition(vec![Zero, Zero]), half),
        ScaledState(ExtendedBasis::Superposition(vec![One, Zero]), half),
        ScaledState(ExtendedBasis::Superposition(vec![Zero, One]), -half),
        ScaledState(ExtendedBasis::Superposition(vec![One, One]), -half),
    ];
    let attempt = basis_to_expand.clone().expand_qubits(expand_both);
    assert!(correct.len() == attempt.len());
    for state in correct {
        assert!(attempt.contains(&state));
    }
}

#[test]
fn test_hadamard_cnot_entanglement() {
    let circuit = Circuit::new(2).h(0).cx(&[0], 1);
    for _ in 0..1000 {
        let mut sim = match SyntaxSimulator::build(circuit.clone()) {
            Ok(sim) => sim,
            Err(e) => panic!("Error building simulator: {}", e),
        };
        sim.run();
        let result = sim.state().collapse();
        assert!(result == 0b00 || result == 0b11);
    }
}

#[test]
fn probability_distribution_sums_to_one() {
    let mut circuit = Circuit::new(2).h(0).cx(&[0], 1);

    for i in 0..10 {
        circuit = circuit.h(0).u(random(), random(), random(), i % 2);
    }

    let mut sim = match SyntaxSimulator::build(circuit) {
        Ok(sim) => sim,
        Err(e) => panic!("Error building simulator: {}", e),
    };
    sim.run();
    let distribution = sim.state();
    let total_probability: f32 = distribution
        .iter()
        .map(|(_, scalar)| scalar.probability())
        .sum();
    assert!(
        (total_probability - 1.0).abs() < 1e-4,
        "Total probability does not sum to 1, got {}",
        total_probability
    );
}

#[test]
fn test_ry_inversion() {
    let circuit = Circuit::new(1).ry(PI, 0);
    for _ in 0..1000 {
        let mut sim = match SyntaxSimulator::build(circuit.clone()) {
            Ok(sim) => sim,
            Err(e) => panic!("Error building simulator: {}", e),
        };
        sim.run();
        let result = sim.state().collapse();
        assert_eq!(result, 1);
    }
}

#[test]
fn test_invert_qubit_through_rz() {
    let circuit = Circuit::new(1).h(0).rz(PI, 0).h(0);
    for _ in 0..1000 {
        let mut sim = match SyntaxSimulator::build(circuit.clone()) {
            Ok(sim) => sim,
            Err(e) => panic!("Error building simulator: {}", e),
        };
        sim.run();
        let result = sim.state().collapse();
        assert_eq!(result, 1);
    }
}

#[test]
fn swap() {
    const N: usize = 10;
    for n_qubits in 1..=N {
        let mut circuit = Circuit::new(n_qubits).h(0).z(0);
        for i in 0..(n_qubits - 1) {
            circuit = circuit.swap(i, i + 1);
        }

        circuit = circuit.h(n_qubits - 1);

        let mut sim = match SyntaxSimulator::build(circuit.clone()) {
            Ok(sim) => sim,
            Err(e) => panic!("Error building simulator: {}", e),
        };
        sim.run();
        let result = sim.state().collapse();
        assert_eq!(result, 1 << (n_qubits - 1));
    }
}

/// The QFT turns the most significant bit into |+⟩ if the parity of the input is even,
/// and into |−⟩ if the parity is odd, so measuring it after hadamard will give 0 for
/// even parity and 1 for odd parity.
#[test]
fn test_parity_by_qft() {
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
        let mut sim = match SyntaxSimulator::build(circuit_with_input.qft(&qft_targets).h(n - 1)) {
            Ok(sim) => sim,
            Err(e) => panic!("Error building simulator: {}", e),
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

#[test]
fn swap_respects_controls() {
    let circuit = Circuit::new(3).x(1).h(0).cswap(&[0], 1, 2);
    for _ in 0..1000 {
        let mut sim = match SyntaxSimulator::build(circuit.clone()) {
            Ok(sim) => sim,
            Err(e) => panic!("Error building simulator: {}", e),
        };
        sim.run();
        let result = sim.state().collapse();
        // If the least significant bit is 0, the swap should not happen and we should get 0b010,
        // but if it's 1, the swap should happen and we should get 0b101.
        assert!(
            result == 0b010 || result == 0b101,
            "Failed with result {:03b}",
            result
        );
    }
}

#[test]
fn qubit_expansion_respects_i() {
    let basis_to_expand = ExtendedBasis::Superposition(vec![ExtendedQubitBasis::I]);

    use ExtendedQubitBasis::*;
    assert_eq!(
        basis_to_expand.clone().expand_qubit(0),
        [
            Some(ScaledState(
                ExtendedBasis::Superposition(vec![Zero]),
                Scalar::FRAC_1_SQRT_2
            )),
            Some(ScaledState(
                ExtendedBasis::Superposition(vec![One]),
                Scalar::I * Scalar::FRAC_1_SQRT_2
            )),
        ]
    );
}

#[test]
fn qubit_expansion_respects_minus() {
    let basis_to_expand = ExtendedBasis::Superposition(vec![ExtendedQubitBasis::Minus]);

    use ExtendedQubitBasis::*;
    assert_eq!(
        basis_to_expand.clone().expand_qubit(0),
        [
            Some(ScaledState(
                ExtendedBasis::Superposition(vec![Zero]),
                Scalar::FRAC_1_SQRT_2
            )),
            Some(ScaledState(
                ExtendedBasis::Superposition(vec![One]),
                -Scalar::FRAC_1_SQRT_2
            )),
        ]
    );
}

#[test]
fn respects_global_phase_against_matrix_multiplication() {
    let circuit = Circuit::new(1).x(0).s(0).h(0);

    let mut syntax_simulator = match SyntaxSimulator::build(circuit.clone()) {
        Ok(simulator) => simulator,
        Err(error) => panic!("Error building simulator: {}", error),
    };
    syntax_simulator.run();

    let mut expected = StateVector::zeros(1);
    for gate in [
        Gate::new(GateType::X, &[], &[0]).unwrap(),
        Gate::new(GateType::S, &[], &[0]).unwrap(),
        Gate::new(GateType::H, &[], &[0]).unwrap(),
    ] {
        expected.apply_matrix(&expand_matrix_from_gate(&gate, 1));
    }

    let mut actual = StateVector::zeros(1);
    let syntax_state = syntax_simulator.state();
    for basis in 0..2 {
        actual[basis] = syntax_state.basis_value(basis).into();
    }

    assert!(equal_state_c(&actual, &expected, 1, 0.001));
}

#[test]
fn matches_matrix_multiplication_for_single_qubit_gate_types() {
    let circuit = Circuit::new(1)
        .x(0)
        .y(0)
        .z(0)
        .h(0)
        .s(0)
        .u(PI / 3.0, PI / 4.0, -PI / 5.0, 0);

    let mut syntax_simulator = match SyntaxSimulator::build(circuit.clone()) {
        Ok(simulator) => simulator,
        Err(error) => panic!("Error building simulator: {}", error),
    };
    syntax_simulator.run();

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

    let mut actual = StateVector::zeros(1);
    let syntax_state = syntax_simulator.state();
    for basis in 0..2 {
        actual[basis] = syntax_state.basis_value(basis).into();
    }

    assert!(equal_state_c(&actual, &expected, 1, 0.001));
}

#[test]
fn matches_matrix_multiplication_for_swap() {
    let circuit = Circuit::new(2).x(0).h(1).swap(0, 1);

    let mut syntax_simulator = match SyntaxSimulator::build(circuit.clone()) {
        Ok(simulator) => simulator,
        Err(error) => panic!("Error building simulator: {}", error),
    };
    syntax_simulator.run();

    let mut expected = StateVector::zeros(2);
    for gate in [
        Gate::new(GateType::X, &[], &[0]).unwrap(),
        Gate::new(GateType::H, &[], &[1]).unwrap(),
        Gate::new(GateType::SWAP, &[], &[0, 1]).unwrap(),
    ] {
        expected.apply_matrix(&expand_matrix_from_gate(&gate, 2));
    }

    let mut actual = StateVector::zeros(2);
    let syntax_state = syntax_simulator.state();
    for basis in 0..4 {
        actual[basis] = syntax_state.basis_value(basis).into();
    }

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

#[test]
fn random_circuits_match_matrix_multiplication() {
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

        let mut syntax_simulator = match SyntaxSimulator::build(circuit) {
            Ok(simulator) => simulator,
            Err(error) => panic!("Error building simulator: {}", error),
        };
        syntax_simulator.run();

        let mut actual = StateVector::zeros(N_QUBITS);
        let syntax_state = syntax_simulator.state();
        for basis in 0..(1 << N_QUBITS) {
            actual[basis] = syntax_state.basis_value(basis).into();
        }

        assert!(
            equal_state_c(&actual, &expected, N_QUBITS, 0.01),
            "Random circuit mismatch on trial {} with seed 0x{:016X}: {:?}",
            trial,
            seed,
            sequence
        );
    }
}
