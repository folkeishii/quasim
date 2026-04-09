use core::{f32, panic};
use std::{
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

#[derive(Debug, Clone, PartialEq)]
enum ExtendedQubitBasis {
    Zero,
    One,
    Plus,
    Minus,
    I,
    MinusI,
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
}

#[derive(Debug, Clone)]
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

impl ExtendedBasis {
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
                trace!("extended basis: {:?}", extended_basis);
                let lesser_bits = Self::all_inherent_states_of(ExtendedBasis::Superposition(
                    extended_basis.clone(),
                ));
                trace!("Lesser bits: {:?}", lesser_bits);
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

#[derive(Debug, Clone)]
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

    // Gates

    pub fn apply_cx(self: &mut SumOfScaledStates, controls: QBits, target: QBits) {
        self.sum = self
            .sum
            .iter()
            .map(|ScaledState(basis, outer_scalar)| {
                let basis = basis.clone();

                // This is bad. We should only get inherent states if the control bits are a superposition.
                // But this currently turns non-control bits that are in superposition into binary bits, which is not ideal.
                let expanded_and_modified_states: Vec<ScaledState> = basis
                    .all_inherent_states()
                    .iter()
                    .map(|ScaledState(true_basis, inner_scalar)| {
                        let term_scalar = *outer_scalar * (*inner_scalar);
                        let binary_basis = true_basis.into_binary(); // Basis should be binary at this point.

                        let new_basis = if binary_basis & controls.get_bitstring()
                            == controls.get_bitstring()
                        {
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
                    .collect();

                expanded_and_modified_states
            })
            .flatten()
            .collect();
        trace!(
            "After applying CX with controls {:?} and target {:?}, state is: {:?}",
            controls, target, self.sum
        );
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
            X => expr.apply_cx(controls, targets),
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
