mod basis;
mod scalar;
mod state;
#[cfg(test)]
mod tests;

use std::{collections::HashMap, convert::Infallible};

use log::debug;
use nalgebra::Complex;

use crate::{
    circuit::{Circuit, CircuitBehaviour, HybridCircuit, pc::CircuitPc},
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{Debuggable, QuantumState, Sampleable, Simulator, StoredCircuit, StoredRegisters},
    syntax_simulator::state::{ScaledState, SumOfScaledStates},
};

pub struct SyntaxSimulator {
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    sum: SumOfScaledStates,
    registers: RegisterFile,
}

impl<B> Sampleable<B> for SyntaxSimulator
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
{
}

impl SyntaxSimulator {
    fn step(&mut self) -> Option<&SumOfScaledStates> {
        let Some(instruction) = self.circuit.instruction(&self.pc) else {
            if self.pc.ret() {
                return Some(&self.sum);
            }
            return None;
        };

        match instruction {
            Instruction::Gate(gate) => {
                self.sum.apply_gate(&gate);
                self.pc.increment();
            }
            Instruction::MeasureBit(target, (reg, bit_pos)) => {
                let measured = self.sum.calculate_probability_distribution().collapse();
                let measured_bit = (measured >> target) & 1;
                self.registers[reg.as_str()]
                    .write_bit(bit_pos, measured_bit)
                    .expect("invalid register write");
                self.sum.collapse_to_binary_state(measured);
                self.pc.increment();
            }
            Instruction::MeasureAll(reg) => {
                let measured = self.sum.calculate_probability_distribution().collapse();
                self.registers[reg.as_str()].write(measured);
                self.sum.collapse_to_binary_state(measured);
                self.pc.increment();
            }
            Instruction::Jump(label_pc) => {
                self.pc.jump(label_pc);
            }
            Instruction::JumpIf(expr, label_pc) => {
                if expr.eval(&self.registers) {
                    self.pc.jump(label_pc);
                } else {
                    self.pc.increment();
                }
            }
            Instruction::Assign(expr, reg) => {
                let value = expr.eval(&self.registers);
                self.registers[reg.as_str()].write(value);
                self.pc.increment();
            }
            Instruction::Call(name, lsq, ctrl) => {
                self.pc.jump_and_link(name, lsq, ctrl);
            }
        }

        #[cfg(debug_assertions)]
        {
            // In debug mode, we can afford to compute state between every step, so we do it
            // at every step to check for consistency and correctness.
            debug!(
                "All scaled states: {:?}",
                self.sum.all_scaled_states().collect::<Vec<_>>()
            );
        }

        Some(&self.sum)
    }

    fn step_all(&mut self) -> &SumOfScaledStates {
        while self.step().is_some() {}
        &self.sum
    }
}

impl<B> TryFrom<Circuit<B>> for SyntaxSimulator
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
{
    type Error = Infallible;

    fn try_from(circuit: Circuit<B>) -> Result<Self, Self::Error> {
        let n_qubits = circuit.n_qubits();
        let circuit: Circuit<HybridCircuit> = circuit.into();
        let registers = RegisterFile::from(circuit.registers());

        Ok(Self {
            circuit,
            pc: CircuitPc::default(),
            sum: SumOfScaledStates::guaranteed_full_zero(n_qubits),
            registers,
        })
    }
}

impl Simulator for SyntaxSimulator {
    type BasisValue = Complex<f32>;
    type State = HashMap<usize, Complex<f32>>;

    fn run(&mut self) {
        self.reset();
        self.step_all();
    }

    fn reset(&mut self) {
        self.pc = CircuitPc::default();
        // Old probability cache gets destroyed when old sum gets destroyed
        self.sum = SumOfScaledStates::guaranteed_full_zero(self.circuit.n_qubits());
        self.registers.reset();
    }

    fn state(&self) -> &Self::State {
        &self.sum.calculate_probability_distribution()
    }
}

impl Debuggable for SyntaxSimulator {
    fn next(&mut self) -> bool {
        self.step().is_some()
    }

    fn double_ended(&self) -> bool {
        false
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<crate::instruction::Instruction>) {
        (&self.pc, self.circuit.instruction(&self.pc))
    }
}

impl StoredCircuit for SyntaxSimulator {
    type B = HybridCircuit;

    fn circuit(&self) -> &Circuit<HybridCircuit> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<HybridCircuit> {
        &mut self.circuit
    }
}

impl StoredRegisters for SyntaxSimulator {
    fn registers(&self) -> &RegisterFile {
        &self.registers
    }
}
