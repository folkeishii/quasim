use core::f32;
use std::{
    collections::HashMap,
    fmt::{Debug, Display, write},
    iter,
    ops::{Mul, Neg},
};

use log::trace;
use nalgebra::{Complex, DVector};
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
type TrueBasis = usize;

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
struct ExtendedBasis(Vec<ExtendedQubitBasis>);

impl From<ScaledQubitBasis> for ScaledState {
    fn from(value: ScaledQubitBasis) -> Self {
        use ExtendedQubitBasis::*;
        let basis = vec![if value.0 { One } else { Zero }];
        ScaledState(ExtendedBasis(basis), value.1)
    }
}

impl ExtendedBasis {
    fn inherent_states(mut extended_basis: ExtendedBasis) -> Vec<(TrueBasis, Scalar)> {
        let Some(basis) = extended_basis.0.pop() else {
            panic!("Cannot get inherent states of empty basis!");
        };

        if extended_basis.0.is_empty() {
            return basis
                .semantic_expressions()
                .iter()
                .filter(|ScaledQubitBasis(_, scalar)| !scalar.is_zero())
                .map(|ScaledQubitBasis(basis, scalar)| {
                    let basis = if *basis { 1 } else { 0 };
                    (basis, *scalar)
                })
                .collect();
        }

        let semantics = basis.semantic_expressions();
        // trace!("Basis: {:?}, Semantic expressions: {:?}", basis, semantics);

        semantics
            .into_iter()
            .filter(|ScaledQubitBasis(_, s)| !s.is_zero())
            .map(|msb| {
                trace!("extended basis: {:?}", extended_basis);
                let lesser_bits = Self::inherent_states(extended_basis.clone());
                trace!("Lesser bits: {:?}", lesser_bits);
                lesser_bits
                    .iter()
                    .map(|lsb| {
                        let basis = (msb.0 as usize) << extended_basis.0.len() | (lsb.0 as usize);
                        let scalar = msb.1 * lsb.1;
                        trace!("Basis: {}, Scalar: {}", basis, scalar);
                        (basis, scalar)
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }
}

#[derive(Debug, Clone)]
struct ScaledState(ExtendedBasis, Scalar);

impl ScaledState {
    fn has_zero_coef(&self) -> bool {
        self.1.is_zero()
    }
}

type Sum = Vec<ScaledState>;

#[derive(Debug, Clone)]
struct SumOfScaledStates {
    n_qubits: usize,
    sum: Vec<ScaledState>,
}

impl SumOfScaledStates {
    fn guaranteed_full_zero(n_qubits: usize) -> Self {
        let mut sum = Vec::<ScaledState>::new();
        let n_zeros: ExtendedBasis =
            ExtendedBasis(iter::repeat_n(ExtendedQubitBasis::Zero, n_qubits).collect());
        sum.push(ScaledState(n_zeros, Scalar::ONE));
        Self { n_qubits, sum }
    }

    fn states(&self) -> impl Iterator<Item = &ExtendedBasis> {
        self.sum.iter().map(|ScaledState(state, _)| state)
    }

    fn probability_distribution(&self) -> impl Iterator<Item = &ScaledState> {
        self.sum.iter()
    }
    /*
    fn evaluate_full_state_vector(&self) -> DVector<Complex<f32>> {
        let mut result: DVector<Complex<f32>> = DVector::zeros(1usize << self.n_qubits);
        for (ScaledState(state, scalar)) in self.sum.iter() {
            let state = state.get_bitstring();
            result[state] = (*scalar).into();
        }
        result
    }
    */
    // Gates
    /*
    pub fn apply_x(self: &mut SumOfScaledStates, controls: QBits, target: QBits) {
        let to_insert: HashMap<QBits, Scalar> = HashMap::new();

        let old_with_controls_set: Vec<(&QBits, &Scalar)> = self
            .sum
            .iter()
            .filter(|(state, _)| {
                **state & controls == controls // && **state & target == QBits::from_bitstring(0)
            })
            .collect();

        for (state, scalar) in old_with_controls_set {
            let state_with_target_flipped = *state ^ target;
            let flipped = match self.sum.get(&state_with_target_flipped) {
                Some(flipped) => {
                    // A flipped variant already exists. Exchange the original and flipped variants.
                    flipped
                }
                None => {
                    // No flipped variant exists. Remove the original and add the flipped variant.
                    continue;
                }
            };
        }
    }
    */
    /*pub fn apply_h(self: &mut SumOfScaledStates, controls: QBits, target: QBits) {
        let mut new_terms: Vec<ScaledState> = Vec::new();
        for term in self.sum.iter_mut() {
            if term.state & controls == controls {
                // Controls are set!
                let flipped_variant = term.state;
                term.scalar.mul_frac_1_sqrt_2()
            }
        }
    }*/
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

        match gate.get_type() {
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
        use ExtendedQubitBasis::*;
        println!(
            "{:?}",
            ExtendedBasis::inherent_states(ExtendedBasis(vec![One, One, Minus, One, One]))
        );

        /*let mut probability_so_far = 0f32;
        let guess = random::<f32>();
        for (state, prob) in self.step_all().probability_distribution() {
            probability_so_far += prob;
            if probability_so_far >= guess {
                return state.get_bitstring() as usize;
            }
        }

        panic!(
            "Probabilities did not sum to 1! Total probability: {}",
            probability_so_far
        );*/
        1337
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
