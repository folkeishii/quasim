use nalgebra::{Complex, DMatrix, DVector};
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::{
    cart,
    circuit::Circuit,
    ext::get_gate_matrix,
    gate::{Gate, QBits},
    instruction::Instruction,
    simulator::{DebuggableSimulator, RunnableSimulator},
};

pub struct SVExecutor<'a> {
    state_vector: DVector<Complex<f32>>,
    circuit: &'a Circuit,
    pc: usize,
}

impl<'a> SVExecutor<'a> {
    /// Step forward one instruction in the circuit
    pub fn step(&mut self) -> Option<&DVector<Complex<f32>>> {
        if self.pc >= self.circuit.instructions().len() {
            return None;
        }

        let inst = &self.circuit.instructions()[self.pc];
        self.apply_instruction(inst);

        self.pc += 1;

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
    pub fn state_vector(&self) -> &DVector<Complex<f32>> {
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
        let u: DMatrix<Complex<f32>> = get_gate_matrix(gate);
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
    }

    fn measure(&mut self, targets: QBits) {
        let measurement = self.get_collapsed_state();
        let mask = targets.get_bitstring();
        let collapsed_bitstring = measurement & mask;

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
            .sum::<f32>()
            .sqrt();
        self.state_vector.iter_mut().for_each(|x| *x /= norm);
    }

    fn apply_instruction(&mut self, inst: &Instruction) {
        match inst {
            Instruction::Gate(gate) => self.gate(gate),
            Instruction::Measurement(qbits) => self.measure(*qbits),
        }
    }
}

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
        self.get_executor().step_all().get_collapsed_state()
    }

    fn final_state(&self) -> DVector<Complex<f32>> {
        self.get_executor().step_all().state_vector().clone()
    }
}

impl SVSimulator {
    pub fn get_executor(&self) -> SVExecutor<'_> {
        let size = 1 << self.circuit.n_qubits();
        let mut init_state_vector: DVector<Complex<f32>> = DVector::from_element(size, cart![0.0]);
        init_state_vector[0] = cart![1.0];

        SVExecutor {
            state_vector: init_state_vector,
            circuit: &self.circuit,
            pc: 0,
        }
    }

    pub fn attach_debugger(&self) -> SVSimulatorDebugger<'_> {
        SVSimulatorDebugger {
            executor: self.get_executor(),
        }
    }
}

pub struct SVSimulatorDebugger<'a> {
    executor: SVExecutor<'a>,
}

impl<'a> DebuggableSimulator for SVSimulatorDebugger<'a> {
    fn next(&mut self) -> Option<&DVector<Complex<f32>>> {
        self.executor.step()
    }

    fn current_instruction(&self) -> Option<(usize, &Instruction)> {
        let pc = self.executor.pc;
        if pc >= self.executor.circuit.instructions().len() {
            return None;
        }
        Some((pc, &self.executor.circuit.instructions()[pc]))
    }

    fn current_state(&self) -> &DVector<Complex<f32>> {
        &self.executor.state_vector
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SVError {}
