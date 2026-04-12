use crate::{
    circuit::Circuit,
    gate::QBits,
    syntax_simulator::{
        basis::{ExtendedBasis, ExtendedQubitBasis},
        scalar::Scalar,
    },
};

use crate::simulator::BuildSimulator;

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
        let result = sim.run();
        assert!(result == 0b00 || result == 0b11);
    }
}
