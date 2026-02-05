use nalgebra::{Complex, DMatrix, DVector};
use crate::{SimpleSimulator, Circuit, Instruction};

pub struct SimpleSimpleSimulator {
    state_vector: Vec<Complex<f32>>,
}

impl SimpleSimulator for SimpleSimpleSimulator {
    type E = SimpleError;
    
    fn build(circuit: crate::Circuit) -> Result<Self, Self::E> {

        let k = circuit.n_qubits;
        let mut init_state_vector = vec![Complex::ZERO; 1 << k];
        init_state_vector[0] = Complex::ONE;

        let mut sim = SimpleSimpleSimulator {
            state_vector: init_state_vector,
        };

        for inst in circuit.instructions {
            sim.apply_instruction(inst);
        }

        Ok(sim)
    }
    
    fn run(&self) -> usize {
        todo!()
    }
    
    fn final_state(&self) -> DVector<Complex<f32>> {
        return DVector::from_vec(self.state_vector.clone());
    }
}

impl SimpleSimpleSimulator {
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

    fn apply_gate(
        &mut self,
        controls: &[usize],
        targets: &[usize],
        u: DMatrix<Complex<f32>>
    )
    {
        // State vector is length 2^n , n=num qubits
        for i in 0..self.state_vector.len() {
            if !SimpleSimpleSimulator::is_block_base(i, targets) {
                continue;
            }

            if !SimpleSimpleSimulator::controls_active(i, controls) {
                continue;
            }

            let indices = SimpleSimpleSimulator::block_indices(i, targets);

            // Read amplitudes
            let v = DVector::from_iterator(
                indices.len(),
                indices.iter().map(|&idx| self.state_vector[idx])
            );

            // Apply gate matrix, assumes u matches size of v
            let v2 = &u * &v;

            // Write updated amplitudes back
            for (j, &idx) in indices.iter().enumerate() {
                self.state_vector[idx] = v2[j];
            }
        }

    }

    fn apply_instruction(&mut self, inst: Instruction) {
        match inst {
            Instruction::X(t) => self.apply_gate(&[], &[t], inst.get_matrix()),
            Instruction::Y(t) => self.apply_gate(&[], &[t], inst.get_matrix()),
            Instruction::Z(t) => self.apply_gate(&[], &[t], inst.get_matrix()),
            Instruction::H(t) => self.apply_gate(&[], &[t], inst.get_matrix()),
            Instruction::CNOT(c, t) => self.apply_gate(&[c], &[t], inst.get_matrix()),
        };
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimpleError {

}