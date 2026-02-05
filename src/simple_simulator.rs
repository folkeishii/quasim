use nalgebra::{Complex, DMatrix, DVector};
use crate::{SimpleSimulator, Circuit, Instruction};

pub struct SimpleSimpleSimulator {
    state_vector: Vec<Complex<f32>>,
    
    /* For now, the *simple* simulator can hold an entire circuit,
     * maybe there is a better way? 
    */
    circuit: Circuit,
}

impl SimpleSimulator for SimpleSimpleSimulator {
    type E = SimpleError;
    
    fn build(circuit: crate::Circuit) -> Result<Self, Self::E> {
        todo!()
    }
    
    fn run(&self) -> usize {
        todo!()
    }
    
    fn final_state(&self) -> nalgebra::DVector<nalgebra::Complex<f32>> {
        todo!()
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

    fn apply_controlled_gate(
        &mut self,
        controls: Vec<usize>,
        targets: Vec<usize>,
        u: DMatrix<Complex<f32>>
    )
    {
        // State vector is length 2^n , n=num qubits
        for i in 0..self.state_vector.len() {
            if !SimpleSimulator::is_block_base(i, &targets) {
                continue;
            }

            if !SimpleSimulator::controls_active(i, &controls) {
                continue;
            }

            let indices = SimpleSimulator::block_indices(i, &targets);

            // Read amplitudes
            let mut v = Vec::with_capacity(indices.len());
            for &idx in &indices {
                v.push(self.state_vector[idx]);
            }

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
}

#[derive(thiserror::Error)]
pub enum SimpleError {

}