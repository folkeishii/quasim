use nalgebra::{Complex, DMatrix, DVector};
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::simulator::HybridSimulator;
use crate::{
    cart,
    circuit::{Circuit, CircuitPc},
    expr_dsl::{Expr, Value},
    ext::get_gate_matrix,
    gate::{Gate, QBits},
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{
        DebuggableSimulator, DoubleEndedSimulator, RunnableSimulator, StoredCircuitSimulator,
    },
};

struct SVExecutor {
    state_vector: DVector<Complex<f64>>,
    circuit: Circuit,
    pc: usize,
    registers: RegisterFile<Value>,
}

impl SVExecutor {
    fn new(circuit: Circuit) -> Self {
        let size = 1 << circuit.n_qubits();
        let mut init_state_vector: DVector<Complex<f64>> = DVector::from_element(size, cart![0.0]);
        init_state_vector[0] = cart![1.0];

        let registers = RegisterFile::from(circuit.registers());
        Self {
            state_vector: init_state_vector,
            circuit: circuit,
            pc: 0,
            registers: registers,
        }
    }

    /// Step forward one instruction in the circuit
    pub fn step(&mut self) -> Option<&DVector<Complex<f64>>> {
        if self.pc >= self.circuit.instructions().len() {
            return None;
        }

        let inst = self.circuit.instructions()[self.pc].clone();
        self.apply_instruction(&inst);

        Some(&self.state_vector)
    }

    /// Run the entire circuit
    pub fn step_all(&mut self) -> &Self {
        while let Some(_) = self.step() {}
        self
    }

    /// Gets a collapsed result from the current state vector
    pub fn get_collapsed_state(&self) -> usize {
        let probs = self.state_vector.iter().map(|&c| c.norm_sqr());

        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }

    /// Get current state of the quantum system
    pub fn state_vector(&self) -> &DVector<Complex<f64>> {
        &self.state_vector
    }

    /// Checks that all control bits are 1
    fn controls_active(i: usize, controls: QBits) -> bool {
        let control_mask = controls.get_bitstring();
        (i & control_mask) == control_mask
    }

    /// Checks that all target bits are 0
    fn is_block_base(i: usize, targets: QBits) -> bool {
        let target_mask = targets.get_bitstring();
        (i & target_mask) == 0
    }

    fn block_indices(base: usize, targets: &[usize]) -> Vec<usize> {
        let k = targets.len();
        let mut indices = Vec::with_capacity(1 << k);

        for mask in 0..(1 << k) {
            let mut idx = base;
            // We embed the bits of mask into the corresponding target qubit positions
            for (j, &target_bit_pos) in targets.iter().enumerate() {
                // Take the j:th bit of the mask and place it at the correct position
                // of the target bit
                idx |= ((mask >> j) & 1) << target_bit_pos;
            }
            indices.push(idx);
        }

        indices
    }

    fn gate(&mut self, gate: &Gate) {
        let controls = gate.get_control_bits();
        let targets = gate.get_target_bits();
        let u: DMatrix<Complex<f64>> = get_gate_matrix(gate);
        let target_indices = targets.get_indices();

        // State vector is length 2^n , n=num qubits
        for i in 0..self.state_vector.len() {
            if !Self::is_block_base(i, targets) {
                continue;
            }

            if !Self::controls_active(i, controls) {
                continue;
            }

            let indices = Self::block_indices(i, &target_indices);

            // Read amplitudes
            let v = DVector::from_iterator(
                indices.len(),
                indices.iter().map(|&idx| self.state_vector[idx]),
            );

            // Apply gate matrix, assumes u matches size of v
            let v2 = &u * &v;

            // Write updated amplitudes back
            for (j, &idx) in indices.iter().enumerate() {
                self.state_vector[idx] = v2[j];
            }
        }

        self.pc += 1;
    }

    fn measure(&mut self, targets: QBits, reg: &str) {
        let measurement = self.get_collapsed_state();
        let mask = targets.get_bitstring();
        let collapsed_bitstring = measurement & mask;

        let mut bits_compacted = 0;
        for (i, bit) in targets.get_indices().into_iter().enumerate() {
            let value = (collapsed_bitstring >> bit) & 1;
            bits_compacted |= value << i;
        }
        self.registers[reg] = Value::Int(bits_compacted as i32);

        // Go through state vector and remove amplitude for all states that do not align with measurement
        for (i, amp) in self.state_vector.iter_mut().enumerate() {
            if (i & mask) != collapsed_bitstring {
                *amp = Complex::ZERO;
            }
        }

        // Renormalize state vector
        let norm = self
            .state_vector
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>()
            .sqrt();
        self.state_vector.iter_mut().for_each(|x| *x /= norm);

        self.pc += 1;
    }

    fn jump(&mut self, label_pc: &CircuitPc) {
        if let Some(_sub_circuit) = label_pc.sub_circuit() {
            todo!("TODO: handle label inside sub circuit")
        } else {
            self.pc = label_pc.pc();
        }
    }

    fn jump_if(&mut self, expr: &Expr, label_pc: &CircuitPc) {
        match expr.eval(&self.registers) {
            Ok(Value::Bool(true)) => self.jump(label_pc),
            Ok(Value::Bool(false)) => self.pc += 1,
            Err(err) => panic!("{}", err),
            _ => panic!(
                "Expression was expected to evaluate to boolean type but got something else."
            ),
        }
    }

    fn assign(&mut self, expr: &Expr, reg: &str) {
        match expr.eval(&self.registers) {
            Ok(value) => self.registers[reg] = value,
            Err(err) => panic!("{}", err),
        }
        self.pc += 1;
    }

    fn apply_instruction(&mut self, inst: &Instruction) {
        match inst {
            Instruction::Gate(gate) => self.gate(gate),
            Instruction::Measurement(qbits, reg) => self.measure(*qbits, reg),
            Instruction::Jump(label_pc) => self.jump(label_pc),
            Instruction::JumpIf(expr, label_pc) => self.jump_if(expr, label_pc),
            Instruction::Assign(expr, reg) => self.assign(expr, reg),
            Instruction::SubCircuit(_, _) => todo!(),
        }
    }
}

// SVSimulator

pub struct SVSimulator {
    circuit: Circuit,
}

impl TryFrom<Circuit> for SVSimulator {
    type Error = SVError;

    fn try_from(value: Circuit) -> Result<Self, Self::Error> {
        Ok(Self { circuit: value })
    }
}

impl RunnableSimulator for SVSimulator {
    fn run(&self) -> usize {
        SVExecutor::new(self.circuit.clone())
            .step_all()
            .get_collapsed_state()
    }

    fn final_state(&self) -> DVector<Complex<f64>> {
        SVExecutor::new(self.circuit.clone())
            .step_all()
            .state_vector()
            .clone()
    }
}

// SVSimulatorDebugger

pub struct SVSimulatorDebugger {
    executor: SVExecutor,
}

impl TryFrom<Circuit> for SVSimulatorDebugger {
    type Error = SVError;

    fn try_from(value: Circuit) -> Result<Self, Self::Error> {
        Ok(Self {
            executor: SVExecutor::new(value),
        })
    }
}

impl DebuggableSimulator for SVSimulatorDebugger {
    fn next(&mut self) -> Option<&DVector<Complex<f64>>> {
        self.executor.step()
    }

    fn current_instruction(&self) -> Option<(usize, &Instruction)> {
        let pc = self.executor.pc;
        if pc >= self.executor.circuit.instructions().len() {
            return None;
        }
        Some((pc, &self.executor.circuit.instructions()[pc]))
    }

    fn current_state(&self) -> &DVector<Complex<f64>> {
        &self.executor.state_vector
    }
}

impl StoredCircuitSimulator for SVSimulatorDebugger {
    fn circuit(&self) -> &Circuit {
        &self.executor.circuit
    }
}

impl DoubleEndedSimulator for SVSimulatorDebugger {
    fn prev(&mut self) -> Option<&DVector<Complex<f64>>> {
        todo!() // For the sake of being able to run debug terminal
    }
}

impl HybridSimulator<Value> for SVSimulatorDebugger {
    fn registers(&self) -> &RegisterFile<Value> {
        &self.executor.registers
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SVError {}

#[cfg(test)]
mod tests {
    use crate::expr_dsl::Value;
    use crate::simulator::HybridSimulator;
    use crate::sv_simulator::SVSimulatorDebugger;
    use crate::{
        circuit::Circuit,
        expr_dsl::expr_helpers::r,
        simulator::{BuildSimulator, RunnableSimulator},
        sv_simulator::SVSimulator,
    };

    #[test]
    fn test() {
        let circuit = Circuit::new(4)
            .new_reg("r0")
            .new_reg("r1")
            .new_reg("r2")
            .new_reg("r3")
            // Init random state
            .hadamard(0)
            .hadamard(1)
            .hadamard(2)
            .hadamard(3)
            .measure_bit(0, "r0")
            .measure_bit(1, "r1")
            .measure_bit(2, "r2")
            .measure_bit(3, "r3")
            .apply_if(r("r0").eq(1))
            .x(0)
            .apply_if(r("r1").eq(1))
            .x(1)
            .apply_if(r("r2").eq(1))
            .x(2)
            .apply_if(r("r3").eq(1))
            .x(3);

        let sim = SVSimulator::build(circuit.clone()).unwrap();

        println!("{}", sim.final_state());
    }

    #[test]
    fn test_register() {
        let circuit = Circuit::new(2).new_reg("r0").x(1).measure_bit(1, "r0");

        let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
        sim.executor.step_all();

        assert_eq!(sim.register("r0"), Value::Int(1));
    }
}
