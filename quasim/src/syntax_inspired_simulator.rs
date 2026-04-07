/*
NOTICE
This was an attempt at storing the expression with a hashmap.
Finding the scalar of a particular state would be faster.
The cost is that applying gates is more complicated, as we cannot
simply change the state of a term, as the state is the key of the hashmap.
Also, the plus, minus, i and -i states are not considered here, which is pretty costly.
*/
use core::f32;
use std::{collections::HashMap, fmt::Display, iter};

use nalgebra::{Complex, DVector};
use rand::random;

use crate::{
    circuit::{Circuit, PureCircuit, pc::CircuitPc},
    gate::{Gate, GateType, QBits},
    instruction::PureInstruction,
};

#[derive(Debug, Clone, PartialEq)]
enum BasisState {
    Zero,
    One,
}

#[derive(Debug, Clone, PartialEq, Copy)]
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

    fn mul_frac_1_sqrt_2(&mut self) {
        self.frac_1_sqrt_2_power += 1;
    }
}

impl Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.frac_1_sqrt_2_power == 0 {
            write!(f, "{}", self.number)
        } else {
            write!(
                f,
                "({} • (1/√2) ^ {}))",
                self.number, self.frac_1_sqrt_2_power
            )
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
/*
#[derive(Debug, Clone, Copy)]
struct ScaledState {
    scalar: Scalar,
    state: QBits,
}

type Sum<'a> = Vec<&'a ScaledState>;
*/
#[derive(Debug, Clone)]
struct SumOfScaledStates {
    n_qubits: usize,
    sum: HashMap<QBits, Scalar>,
}

impl SumOfScaledStates {
    fn guaranteed_full_zero(n_qubits: usize) -> Self {
        let mut sum = HashMap::new();
        sum.insert(QBits::from_bitstring(0), Scalar::ONE);
        Self { n_qubits, sum }
    }

    fn states(&self) -> impl Iterator<Item = &QBits> {
        self.sum.keys().into_iter()
    }

    fn probability_distribution(&self) -> impl Iterator<Item = (QBits, f32)> {
        self.sum.iter().map(|(state, scalar)| {
            let scalar: Complex<f32> = (*scalar).into();
            let prob = scalar.norm_sqr();
            (*state, prob)
        })
    }

    fn evaluate_full_state_vector(&self) -> DVector<Complex<f32>> {
        let mut result: DVector<Complex<f32>> = DVector::zeros(1usize << self.n_qubits);
        for (state, scalar) in self.sum.iter() {
            let state = state.get_bitstring();
            result[state] = (*scalar).into();
        }
        result
    }

    // Gates

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

pub struct SyntaxInspiredSimulator {
    circuit: Circuit<PureCircuit>,
    pc: CircuitPc,
    state: SumOfScaledStates,
}

#[derive(Debug, thiserror::Error)]
pub enum SyntaxSimError {
    #[error("Tried to build syntax simulator with non-pure circuit")]
    NonPureCircuit,
}

impl SyntaxInspiredSimulator {
    fn apply_gate(expr: &mut SumOfScaledStates, gate: &Gate) {
        let controls = gate.get_control_bits();
        let targets = gate.get_target_bits();
        assert_eq!(targets.get_indices().len(), 1);

        match gate.get_type() {
            GateType::X => expr.apply_x(controls, targets),
            /*GateType::Y => self.apply_y(i, targets),
            GateType::Z => self.apply_z(i, targets),
            GateType::H => self.apply_h(i, targets),
            GateType::S => self.apply_s(i, targets),
            GateType::SWAP => self.apply_swap(i, targets),
            GateType::U(theta, phi, lambda) => {
                self.apply_unitary2(i, &get_u_matrix2(theta, phi, lambda), targets)
            }*/
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
        let mut probability_so_far = 0f32;
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
        );
    }
}

impl TryFrom<Circuit<PureCircuit>> for SyntaxInspiredSimulator {
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
