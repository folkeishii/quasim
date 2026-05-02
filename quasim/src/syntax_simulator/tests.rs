use std::f32::consts::PI;

use rand::random;

use crate::{
    circuit::Circuit,
    gate::QBits,
    simulator::{Buildable, QuantumState},
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
