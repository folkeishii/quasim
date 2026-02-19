use crate::{
    cart,
    circuit::Circuit,
    instruction::Instruction,
    simulator::{BuildSimulator, RunnableSimulator},
};
use nalgebra::{Complex, DMatrix, DVector};
use rand::{distr::weighted::WeightedIndex, prelude::*};

pub struct SimpleSimulator {
    state_vector: Vec<Complex<f32>>,
}

impl TryFrom<Circuit> for SimpleSimulator {
    type Error = SimpleError;

    fn try_from(value: Circuit) -> Result<Self, Self::Error> {
        let circuit = value;
        let k = circuit.n_qubits();
        let mut init_state_vector = vec![cart!(0.0); 1 << k];
        init_state_vector[0] = cart!(1.0);

        let mut sim = SimpleSimulator {
            state_vector: init_state_vector,
        };

        for inst in circuit.instructions() {
            sim.apply_instruction(inst);
        }

        Ok(sim)
    }
}

impl RunnableSimulator for SimpleSimulator {
    fn run(&self) -> usize {
        let probs = self.state_vector.iter().map(|&c| c.norm_sqr());

        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }

    fn final_state(&self) -> DVector<Complex<f32>> {
        return DVector::from_vec(self.state_vector.clone());
    }
}

impl SimpleSimulator {
    fn controls_active(i: usize, controls: &[usize]) -> bool {
        controls.iter().all(|&c| ((i >> c) & 1) == 1)
    }

    fn is_block_base(i: usize, targets: &[usize]) -> bool {
        targets.iter().all(|&t| ((i >> t) & 1) == 0)
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

    fn apply_gate(&mut self, controls: &[usize], targets: &[usize], u: DMatrix<Complex<f32>>) {
        // State vector is length 2^n , n=num qubits
        for i in 0..self.state_vector.len() {
            if !SimpleSimulator::is_block_base(i, targets) {
                continue;
            }

            if !SimpleSimulator::controls_active(i, controls) {
                continue;
            }

            let indices = SimpleSimulator::block_indices(i, targets);

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

    fn apply_instruction(&mut self, inst: &Instruction) {
        self.apply_gate(&inst.get_controls(), &inst.get_targets(), inst.get_matrix());
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimpleError {
    #[error("Measurement mid-circuit")]
    MidCircuitMeasurement,
}

#[cfg(test)]
mod tests {
    use crate::{
        circuit::Circuit,
        instruction::Instruction,
        simple_simulator::SimpleSimulator,
        simulator::{BuildSimulator, RunnableSimulator},
    };

    #[test]
    fn state_vector_print() {
        let circ = Circuit::from_instructions(
            3,
            vec![
                Instruction::H(0),
                Instruction::CNOT(0, 2),
                Instruction::X(0),
                Instruction::H(0),
                Instruction::Y(1),
            ],
        );

        let sim = SimpleSimulator::build(circ).unwrap();
        println!("{}", sim.final_state());
        println!("{:03b}", sim.run());
    }

    #[test]
    fn bell_state_test() {
        let circ = Circuit::from_instructions(2, vec![Instruction::H(0), Instruction::CNOT(0, 1)]);

        let sim = SimpleSimulator::build(circ).unwrap();
        println!("{:02b}", sim.run());
    }

    #[test]
    fn swap_gate_test() {
        let circ = Circuit::from_instructions(2, vec![Instruction::X(1), Instruction::SWAP(0, 1)]);

        let sim = SimpleSimulator::build(circ).unwrap();
        println!("{}", sim.final_state());
        println!("{:02b}", sim.run());
    }
}
