use nalgebra::{Complex, DMatrix, DVector, Normed};
use rand::{distr::weighted::WeightedIndex, prelude::*};
use crate::{SimpleSimulator, Debugger, Circuit, Instruction};

pub struct SimpleSimpleSimulator {
    circuit: Circuit,
}

impl Debugger for SimpleSimpleSimulator {
    
}


impl SimpleSimulator for SimpleSimpleSimulator {
    type E = SimpleError;
    
    fn build(circuit: Circuit) -> Result<Self, Self::E> {
        let sim = SimpleSimpleSimulator {
            circuit: circuit
        };

        Ok(sim)
    }
    
    fn run(&self) -> usize {
        let k = self.circuit.n_qubits;
        let mut state_vector: Vec<Complex<f32>> = vec![Complex::ZERO; 1 << k];
        state_vector[0] = Complex::ONE;

        let instructions = self.circuit.instructions.clone();
        for inst in self.circuit.instructions {
            self.apply_instruction(inst);
        }
        
        self.get_collapsed_state()
    }
    
    fn final_state(&self) -> DVector<Complex<f32>> {
        return DVector::from_vec(self.state_vector.clone());
    }
}

impl SimpleSimpleSimulator {
    // This function should probably return something indicating which bits collapsed to what
    // For like debugging purposes ig
    fn measure(&mut self, targets: &[usize]) {
        let measurement = self.get_collapsed_state();
        //let collapsed_states: Vec<usize> = targets.iter().map(|&t| (measurement >> t) & 1).collect();

        // Mask is bitstring with 1:s at target position
        let mut mask = 0;
        for t in targets {
            mask |= 1 << t;
        }
        let collapsed_bitstring = measurement & mask;

        // Go through state vector and remove amplitude for all states that do not align with measurement
        for (i, amp) in self.state_vector.iter_mut().enumerate() {
            if (i & mask) != collapsed_bitstring {
                *amp = Complex::ZERO;
            }
        }

        // Renormalize state vector
        let norm = self.state_vector.iter().map(|x| x.norm_sqr()).sum::<f32>().sqrt();
        self.state_vector.iter_mut().for_each(|x| *x /= norm);
    }
    
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

    fn apply_instruction(&mut self, inst: &Instruction) {
        self.apply_gate(&inst.get_controls(), &inst.get_targets(), inst.get_matrix());
    }

    /// Gets a collapsed result from the current state vector
    fn get_collapsed_state(&self) -> usize {
        let probs = self.state_vector.iter().map(|&c| c.norm_sqr());

        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimpleError {
    #[error("Measurement mid-circuit")]
    MidCircuitMeasurement
}


#[cfg(test)]
mod tests {
    use crate::{Circuit, Instruction, SimpleSimpleSimulator, SimpleSimulator};

    #[test]
    fn state_vector_print() {
        let circ = Circuit {
            instructions: vec![
                Instruction::H(0),
                Instruction::CNOT(0, 2),
                Instruction::X(0),
                Instruction::H(0),
                Instruction::Y(1),
            ],
            n_qubits: 3,
        };

        let mut sim = SimpleSimpleSimulator::build(circ).unwrap();
        println!("{}", sim.final_state());
        println!("{:03b}", sim.run());
    }

    #[test]
    fn bell_state_test() {
        let circ = Circuit {
            instructions: vec![
                Instruction::H(0),
                Instruction::CNOT(0, 1),
            ],
            n_qubits: 2,
        };

        let sim = SimpleSimpleSimulator::build(circ).unwrap();
        println!("{:02b}", sim.run());
    }

    #[test]
    fn swap_gate_test() {
        let circ = Circuit {
            instructions: vec![
                Instruction::X(1),
                Instruction::SWAP(0, 1),
            ],
            n_qubits: 2,
        };

        let sim = SimpleSimpleSimulator::build(circ).unwrap();
        println!("{}", sim.final_state());
        println!("{:02b}", sim.run());
    }

    #[test]
    fn swap_gate_test() {
        let circ = Circuit {
            instructions: vec![
                Instruction::X(1),
                Instruction::SWAP(0, 1),
            ],
            n_qubits: 2,
        };

        let sim = SimpleSimpleSimulator::build(circ).unwrap();
        println!("{}", sim.final_state());
        println!("{:02b}", sim.run());
    }
}