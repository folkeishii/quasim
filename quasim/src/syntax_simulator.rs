use core::{f32, panic};
use std::{
    collections::VecDeque,
    fmt::{Debug, Display},
    ops::{Mul, Neg},
    vec,
};

use log::trace;
use nalgebra::Complex;
use rand::random;

use crate::{
    circuit::{Circuit, PureCircuit, pc::CircuitPc},
    gate::{Gate, GateType, QBits},
    instruction::PureInstruction,
};

#[derive(Clone, PartialEq, Copy)]
struct Scalar {
    frac_1_sqrt_2_power: usize,
    number: Complex<f32>,
}

impl Scalar {
    const ZERO: Scalar = Scalar {
        frac_1_sqrt_2_power: 0,
        number: Complex::new(0.0, 0.0),
    };

    const ONE: Scalar = Scalar {
        frac_1_sqrt_2_power: 0,
        number: Complex::new(1.0, 0.0),
    };

    const I: Scalar = Scalar {
        frac_1_sqrt_2_power: 0,
        number: Complex::new(0.0, 1.0),
    };

    const FRAC_1_SQRT_2: Scalar = Scalar {
        frac_1_sqrt_2_power: 1,
        number: Complex::new(1.0, 0.0),
    };

    fn mul_frac_1_sqrt_2(&mut self) {
        self.frac_1_sqrt_2_power += 1;
    }

    fn is_zero(&self) -> bool {
        self == &Scalar::ZERO
    }

    fn probability(&self) -> f32 {
        let c: Complex<f32> = (*self).into();
        c.norm_sqr()
    }
}

impl Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            frac_1_sqrt_2_power: self.frac_1_sqrt_2_power,
            number: -self.number,
        }
    }
}

impl Mul for Scalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            frac_1_sqrt_2_power: self.frac_1_sqrt_2_power + rhs.frac_1_sqrt_2_power,
            number: self.number * rhs.number,
        }
    }
}

impl Mul<ScaledState> for Scalar {
    type Output = ScaledState;

    fn mul(self, rhs: ScaledState) -> Self::Output {
        let ScaledState(state, scalar) = rhs;
        ScaledState(state, self * scalar)
    }
}

impl Mul<Sum> for Scalar {
    type Output = Sum;

    fn mul(self, rhs: Sum) -> Sum {
        rhs.into_iter().map(|ss| self * ss).collect()
    }
}

impl Debug for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.frac_1_sqrt_2_power == 0 {
            write!(f, "{}", self.number)
        } else {
            write!(f, "({} / √2^{}))", self.number, self.frac_1_sqrt_2_power)
        }
    }
}

impl From<Complex<f32>> for Scalar {
    fn from(value: Complex<f32>) -> Self {
        Scalar {
            frac_1_sqrt_2_power: 0,
            number: value,
        }
    }
}

impl Into<Complex<f32>> for Scalar {
    fn into(self) -> Complex<f32> {
        f32::consts::FRAC_1_SQRT_2.powi(self.frac_1_sqrt_2_power as i32) * self.number
    }
}

type TrueQubitBasis = bool;

#[derive(Copy, Clone)]
struct ScaledQubitBasis(TrueQubitBasis, Scalar);

type ScaledQubit = [ScaledQubitBasis; 2];

impl Debug for ScaledQubitBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}|{}⟩", self.1, if self.0 { 1 } else { 0 })
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ExtendedQubitBasis {
    Zero,
    One,
    Plus,
    Minus,
    I,
    MinusI,
}

impl Debug for ExtendedQubitBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ExtendedQubitBasis::*;
        let s = match self {
            Zero => "0",
            One => "1",
            Plus => "+",
            Minus => "-",
            I => "i",
            MinusI => "(-i)",
        };
        write!(f, "{}", s)
    }
}

impl ExtendedQubitBasis {
    fn semantic_expressions(&self) -> ScaledQubit {
        use ExtendedQubitBasis::*;
        match self {
            Zero => [
                ScaledQubitBasis(false, Scalar::ONE),
                ScaledQubitBasis(false, Scalar::ZERO),
            ],
            One => [
                ScaledQubitBasis(true, Scalar::ONE),
                ScaledQubitBasis(true, Scalar::ZERO),
            ],
            Plus => [
                ScaledQubitBasis(false, Scalar::FRAC_1_SQRT_2),
                ScaledQubitBasis(true, Scalar::FRAC_1_SQRT_2),
            ],
            Minus => [
                ScaledQubitBasis(false, Scalar::FRAC_1_SQRT_2),
                ScaledQubitBasis(true, -Scalar::FRAC_1_SQRT_2),
            ],
            I => [
                ScaledQubitBasis(false, Scalar::FRAC_1_SQRT_2),
                ScaledQubitBasis(true, Scalar::FRAC_1_SQRT_2 * Scalar::I),
            ],
            MinusI => [
                ScaledQubitBasis(false, Scalar::FRAC_1_SQRT_2),
                ScaledQubitBasis(true, -Scalar::FRAC_1_SQRT_2 * Scalar::I),
            ],
        }
    }

    fn x(self) -> Self {
        use ExtendedQubitBasis::*;
        match self {
            Zero => One,
            One => Zero,
            Plus => Plus,
            Minus => Minus,
            I => MinusI,
            MinusI => I,
        }
    }
}

#[derive(Clone, PartialEq)]
enum ExtendedBasis {
    Binary(usize),
    Superposition(Vec<ExtendedQubitBasis>),
}

impl ExtendedBasis {
    /// If the basis is binary, checks if the control are satisfied.
    /// If the basis is a superposition, returns None.
    fn check_controls(&self, controls: QBits) -> Option<bool> {
        let controls = controls.get_bitstring();
        if controls == 0 {
            // There are no controls to fail
            return Some(true);
        }

        match self {
            ExtendedBasis::Binary(bitstring) => {
                trace!(
                    "Checking if controls {:?}, are zero for binary basis {}",
                    controls, bitstring
                );
                return Some((bitstring & controls) == controls);
            }
            ExtendedBasis::Superposition(_) => None,
        }
    }

    /// Expands the given qubit in the basis, returning the two cases of the expansion if
    /// the qubit is in an extended basis (eg, |+⟩, |−⟩, |i⟩, |−i⟩),
    /// otherwise returning the same basis if the qubit is already binary (eg, |0⟩, |1⟩).
    fn expand_qubit(self, qubit_to_expand: usize) -> [Option<ScaledState>; 2] {
        use ExtendedBasis::*;
        match self {
            Binary(_) => [Some(ScaledState(self, Scalar::ONE)), None],
            Superposition(bases) => {
                let qubit_basis_to_expand = bases
                    .get(qubit_to_expand)
                    .expect("Tried to expand a qubit out of bounds");

                use ExtendedQubitBasis::*;
                match *qubit_basis_to_expand {
                    One | Zero => {
                        // If the qubit is already in a binary state, there is nothing to expand.
                        return [
                            Some(ScaledState(Self::Superposition(bases), Scalar::ONE)),
                            None,
                        ];
                    }
                    _ => (),
                };

                let apply_minus: bool = matches!(
                    qubit_basis_to_expand,
                    ExtendedQubitBasis::Minus | ExtendedQubitBasis::MinusI
                );

                let mut zero_case = bases.clone();
                zero_case[qubit_to_expand] = Zero;
                let mut one_case = bases;
                one_case[qubit_to_expand] = One;

                [
                    Some(ScaledState(
                        Self::Superposition(zero_case),
                        Scalar::FRAC_1_SQRT_2,
                    )),
                    Some(ScaledState(
                        Self::Superposition(one_case),
                        if apply_minus {
                            -Scalar::FRAC_1_SQRT_2
                        } else {
                            Scalar::FRAC_1_SQRT_2
                        },
                    )),
                ]
            }
        }
    }

    /// Expands all the qubit indexes in the given list, meaning that qubits in an extended
    /// basis (eg, |+⟩, |−⟩, |i⟩, |−i⟩) will be expanded into a sum of binary states,
    /// while qubits already in a binary basis (eg, |0⟩, |1⟩) will be left unchanged.
    fn expand_qubits(self, mut qubits_to_expand: VecDeque<usize>) -> Sum {
        let Some(qubit_to_expand) = qubits_to_expand.pop_front() else {
            return vec![ScaledState(self, Scalar::ONE)];
        };

        use ExtendedBasis::*;
        match self {
            Binary(_) => vec![ScaledState(self, Scalar::ONE)],
            Superposition(_) => {
                // Expand the first qubit, which yields just two terms.
                let possible_expansion = self.clone().expand_qubit(qubit_to_expand);

                // Actually, expanding |0⟩ or |1⟩ just gives one term, so we need to filter out the None case.
                let just_existing_expansions = possible_expansion
                    .iter()
                    // Flatten removes the Option wrapping
                    .flatten();

                // At this point we have all the expansions for the first qubit to expand.
                // Take those terms we expanded into, and just run this function on those terms,
                // but now with one less qubit to expand.

                let all_terms = just_existing_expansions
                    .flat_map(|ScaledState(_, scalar)| {
                        // If a scaled state turns into more terms, the scalar needs to be distributed across the terms.
                        *scalar * self.clone().expand_qubits(qubits_to_expand.clone())
                    })
                    .collect::<Sum>();

                all_terms
            }
        }
    }

    fn try_into_binary(&self) -> Option<usize> {
        match self {
            ExtendedBasis::Binary(bitstring) => Some(*bitstring),
            ExtendedBasis::Superposition(_) => None,
        }
    }

    fn into_binary(&self) -> usize {
        self.try_into_binary()
            .expect("Cannot convert superposition basis into binary basis!")
    }

    fn vec_form(self) -> Vec<ExtendedQubitBasis> {
        match self {
            ExtendedBasis::Binary(mut bitstring) => {
                let mut bases = Vec::new();
                while bitstring > 0 {
                    bases.push(ExtendedQubitBasis::One);
                    bitstring -= 1;
                }
                bases
            }
            ExtendedBasis::Superposition(bases) => bases,
        }
    }

    fn vec_form_padded_to_len(self, n: usize) -> Vec<ExtendedQubitBasis> {
        let mut vec_form = self.vec_form();
        vec_form.resize(n, ExtendedQubitBasis::Zero);
        vec_form
    }

    fn all_inherent_states_of(extended_basis: ExtendedBasis) -> Sum {
        let mut extended_basis = match extended_basis {
            ExtendedBasis::Binary(bitstring) => {
                return vec![ScaledState(ExtendedBasis::Binary(bitstring), Scalar::ONE)];
            }
            ExtendedBasis::Superposition(b) => b,
        };

        let Some(qubit_basis) = extended_basis.pop() else {
            panic!("Cannot get inherent states of empty basis!");
        };

        if extended_basis.is_empty() {
            return qubit_basis
                .semantic_expressions()
                .iter()
                .filter(|ScaledQubitBasis(_, scalar)| !scalar.is_zero())
                .map(|ScaledQubitBasis(basis, scalar)| {
                    let basis = if *basis { 1 } else { 0 };
                    ScaledState(ExtendedBasis::Binary(basis), *scalar)
                })
                .collect();
        }

        let semantics = qubit_basis.semantic_expressions();

        semantics
            .into_iter()
            .filter(|ScaledQubitBasis(_, s)| !s.is_zero())
            .map(|msb| {
                let lesser_bits = Self::all_inherent_states_of(ExtendedBasis::Superposition(
                    extended_basis.clone(),
                ));
                lesser_bits
                    .iter()
                    .map(|lsb| {
                        let lsb_state = match &lsb.0 {
                            ExtendedBasis::Binary(bitstring) => bitstring,
                            ExtendedBasis::Superposition(_) => {
                                panic!("Lesser bits should always be binary at this point!")
                            }
                        };
                        let basis = (msb.0 as usize) << extended_basis.len() | lsb_state;
                        let scalar_product = msb.1 * lsb.1;
                        trace!("Basis: {}, Scalar: {}", basis, scalar_product);
                        ScaledState(ExtendedBasis::Binary(basis), scalar_product)
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }

    pub fn all_inherent_states(self: ExtendedBasis) -> Sum {
        Self::all_inherent_states_of(self.clone())
    }
}

impl From<ScaledQubitBasis> for ScaledState {
    fn from(value: ScaledQubitBasis) -> Self {
        let basis = match value.0 {
            false => 0,
            true => 1,
        };
        ScaledState(ExtendedBasis::Binary(basis), value.1)
    }
}

impl Debug for ExtendedBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtendedBasis::Binary(bitstring) => write!(f, "|{}⟩", bitstring),
            ExtendedBasis::Superposition(bases) => {
                write!(f, "|")?;
                for basis in bases {
                    write!(f, "{:?}", basis)?;
                }
                write!(f, "⟩")?;
                Ok(())
            }
        }
    }
}

#[derive(Clone, PartialEq)]
struct ScaledState(ExtendedBasis, Scalar);

impl ScaledState {
    fn has_zero_coef(&self) -> bool {
        self.1.is_zero()
    }

    fn all_inherent_states(&self) -> Sum {
        let my_scalar = self.1;

        self.0
            .clone()
            .all_inherent_states()
            .iter()
            // Guaranteed usize means cheap clone.
            .map(|substate| my_scalar * substate.clone())
            .collect()
    }

    // Gates

    fn cx(ScaledState(basis, scalar): Self, controls: QBits, target: QBits) -> Vec<ScaledState> {
        if let Some(controls_are_satisfied) = basis.check_controls(controls) {
            if controls_are_satisfied {
                let target_index = *target
                    .get_indices()
                    .first()
                    .expect("Gates are expected to target single qubits!");
                let mut basis = basis.vec_form_padded_to_len(target_index + 1);

                // Flip the target bit in the basis.
                let qubit_basis_to_flip = basis
                    .get_mut(target_index)
                    .expect("Basis should have been padded to include this index.");

                *qubit_basis_to_flip = qubit_basis_to_flip.x();

                return vec![ScaledState(ExtendedBasis::Superposition(basis), scalar)];
            } else {
                return vec![ScaledState(basis, scalar)];
            }
        }
        // At this point, the controls are in superposition...

        // This is bad. We should only get inherent states if the control bits are a superposition.
        // But this currently turns non-control bits that are in superposition into binary bits, which is not ideal.
        basis
            .all_inherent_states()
            .iter()
            .map(|ScaledState(true_basis, inner_scalar)| {
                let term_scalar = scalar * (*inner_scalar);
                let binary_basis = true_basis.into_binary(); // Basis should be binary at this point.

                let new_basis =
                    if binary_basis & controls.get_bitstring() == controls.get_bitstring() {
                        trace!(
                            "Flipping target bit for basis {}, scalar {}",
                            binary_basis, term_scalar
                        );
                        ExtendedBasis::Binary(binary_basis ^ target.get_bitstring())
                    } else {
                        trace!(
                            "Controls not satisfied for basis {} and controls {:?}, scalar {}.",
                            binary_basis, controls, term_scalar
                        );
                        ExtendedBasis::Binary(binary_basis)
                    };
                ScaledState(new_basis, term_scalar)
            })
            .collect()
    }
}

impl Debug for ScaledState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ScaledState(basis, scalar) = self;
        write!(f, "{}|{:?}⟩", scalar, basis)
    }
}

impl From<(usize, Scalar)> for ScaledState {
    fn from(value: (usize, Scalar)) -> Self {
        let basis = value.0;
        let scalar = value.1;
        ScaledState(ExtendedBasis::Binary(basis), scalar)
    }
}

type Sum = Vec<ScaledState>;

#[derive(Debug, Clone)]
struct SumOfScaledStates {
    n_qubits: usize,
    sum: Sum,
}

impl SumOfScaledStates {
    fn guaranteed_full_zero(n_qubits: usize) -> Self {
        let n_zeros = ScaledState(ExtendedBasis::Binary(0), Scalar::ONE);
        Self {
            n_qubits,
            sum: vec![n_zeros],
        }
    }

    fn states(&self) -> impl Iterator<Item = &ExtendedBasis> {
        self.sum.iter().map(|ScaledState(state, _)| state)
    }

    fn probability_distribution(&self) -> impl Iterator<Item = &ScaledState> {
        self.sum.iter()
    }

    pub fn apply_on_all_terms(
        self: &mut SumOfScaledStates,
        cx_one_state: fn(ScaledState, QBits, QBits) -> Vec<ScaledState>,
        controls: QBits,
        target: QBits,
    ) {
        self.sum = self
            .sum
            .iter()
            .map(|ss| cx_one_state(ss.clone(), controls, target))
            .flatten()
            .collect();
    }
}

pub struct SyntaxSimulator {
    circuit: Circuit<PureCircuit>,
    pc: CircuitPc,
    state: SumOfScaledStates,
}

#[derive(Debug, thiserror::Error)]
pub enum SyntaxSimError {
    #[error("Tried to build syntax simulator with non-pure circuit")]
    NonPureCircuit,
}

impl SyntaxSimulator {
    fn apply_gate(expr: &mut SumOfScaledStates, gate: &Gate) {
        let controls = gate.get_control_bits();
        let targets = gate.get_target_bits();
        assert_eq!(targets.get_indices().len(), 1);

        use GateType::*;
        match gate.get_type() {
            X => expr.apply_on_all_terms(ScaledState::cx, controls, targets),
            _ => todo!(),
        }
    }

    fn step(&mut self) -> Option<&SumOfScaledStates> {
        let Some(instruction) = self.circuit.instruction(&self.pc) else {
            return None;
        };
        match instruction {
            PureInstruction::Gate(gate) => {
                Self::apply_gate(&mut self.state, &gate);
                self.pc.increment();
            }
            PureInstruction::Call(_, _, _) => {
                todo!();
            }
        }
        return Some(&self.state);
    }

    fn step_all(&mut self) -> &SumOfScaledStates {
        while self.step().is_some() {}
        &self.state
    }

    pub fn run(&mut self) -> usize {
        self.step_all();

        let mut probability_so_far = 0f32;
        let guess = random::<f32>();
        for ScaledState(state, scalar) in self
            .state
            .probability_distribution()
            .flat_map(|s| s.all_inherent_states())
        {
            probability_so_far += scalar.probability();
            if probability_so_far >= guess {
                return state.into_binary();
            }
        }

        panic!(
            "Probabilities did not sum to 1! Total probability: {}",
            probability_so_far
        );
    }
}

impl TryFrom<Circuit<PureCircuit>> for SyntaxSimulator {
    type Error = SyntaxSimError;

    fn try_from(circuit: Circuit<PureCircuit>) -> Result<Self, Self::Error> {
        let n_qubits = circuit.n_qubits();

        Ok(Self {
            circuit,
            pc: CircuitPc::default(),
            state: SumOfScaledStates::guaranteed_full_zero(n_qubits),
        })
    }
}

#[cfg(test)]
mod tests {
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

        /*let expand_both = QBits::from_indices(&[0, 1]);
        let half = Scalar::FRAC_1_SQRT_2 * Scalar::FRAC_1_SQRT_2;
        assert_eq!(
            basis_to_expand.clone().expand_qubits(expand_both),
            vec![
                ScaledState(ExtendedBasis::Binary(0), half),
                ScaledState(ExtendedBasis::Binary(1), -half),
                ScaledState(ExtendedBasis::Binary(2), half),
                ScaledState(ExtendedBasis::Binary(3), -half),
            ]
        );*/
    }
}
