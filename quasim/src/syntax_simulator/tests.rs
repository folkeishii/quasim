use crate::{
    common_test,
    gate::QBits,
    syntax_simulator::SyntaxSimulator,
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
fn double_sub() {
    common_test::double_sub::<SyntaxSimulator>();
}

#[test]
fn deep_sub() {
    common_test::deep_sub::<SyntaxSimulator>();
}

#[test]
fn hybrid_test() {
    common_test::hybrid_test::<SyntaxSimulator>();
}

#[test]
fn register_test() {
    common_test::register_test::<SyntaxSimulator>();
}

#[test]
fn test_measure_overwrites_with_zero() {
    common_test::test_measure_overwrites_with_zero::<SyntaxSimulator>();
}

#[test]
fn test_reset() {
    common_test::test_reset::<SyntaxSimulator>();
}

#[test]
fn test_reset_with_shared_scratch_register() {
    common_test::test_reset_with_shared_scratch_register::<SyntaxSimulator>();
}

#[test]
fn deep_ctrl_sub() {
    common_test::deep_ctrl_sub::<SyntaxSimulator>();
}

#[test]
fn mid_measure_all() {
    common_test::mid_measure_all::<SyntaxSimulator>();
}

#[test]
fn mid_measure_bit() {
    common_test::mid_measure_bit::<SyntaxSimulator>();
}

#[test]
fn interleaved() {
    common_test::interleaved::<SyntaxSimulator>();
}

#[test]
fn test_hadamard_cnot_entanglement() {
    common_test::hadamard_cnot_entanglement_pure::<SyntaxSimulator>();
}

#[test]
fn probability_distribution_sums_to_one() {
    common_test::probability_distribution_sums_to_one_pure::<SyntaxSimulator>();
}

#[test]
fn test_ry_inversion() {
    common_test::ry_inversion_pure::<SyntaxSimulator>();
}

#[test]
fn test_invert_qubit_through_rz() {
    common_test::invert_qubit_through_rz_pure::<SyntaxSimulator>();
}

#[test]
fn swap() {
    common_test::swap_pure::<SyntaxSimulator>();
}

#[test]
fn test_parity_by_qft() {
    common_test::parity_by_qft_pure::<SyntaxSimulator>();
}

#[test]
fn swap_respects_controls() {
    common_test::swap_respects_controls_pure::<SyntaxSimulator>();
}

#[test]
fn respects_global_phase_against_matrix_multiplication() {
    common_test::respects_global_phase_against_matrix_multiplication_pure::<SyntaxSimulator>();
}

#[test]
fn matches_matrix_multiplication_for_single_qubit_gate_types() {
    common_test::matches_matrix_multiplication_for_single_qubit_gate_types_pure::<SyntaxSimulator>(
    );
}

#[test]
fn matches_matrix_multiplication_for_swap() {
    common_test::matches_matrix_multiplication_for_swap_pure::<SyntaxSimulator>();
}

#[test]
fn random_circuits_match_matrix_multiplication() {
    common_test::random_circuits_match_matrix_multiplication_pure::<SyntaxSimulator>();
}
