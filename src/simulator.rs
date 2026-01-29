use nalgebra::{Complex, DVector};
use crate::Circuit;
use rand::prelude::*;

pub struct Qubit {
    alfa: Complex<f32>,
    beta: Complex<f32>,
}

impl Default for Qubit {
    // Default qubit is a guaranteed 0 with no phase AKA |0>
    fn default() -> Self { 
        {Complex::new(1, 0), Complex::new(0)} 
    } 
}

pub struct SimpleSimulator {
    /* The state is the vector of complex numbers indicating the
     * phase of, and probability of measuring, each representable 
     * n-bit word from the n-qubit computer.
     * Instead of storing the giant state vector, we chose to store
     * a vector of each qubit for now. It should be possible to map
     * between them...
    */
    qubits: Vec<Qubit>,
    
    /* For now, the *simple* simulator can hold an entire circuit,
     * maybe there is a better way? 
    */
    circuit: Circuit,
}

pub struct Collapsed(usize);

impl Collapsed {
    fn get_bit(self: &Self, n: usize) -> bool {
        (self.0 & (1 << n)) > 0
    }
}

impl SimpleSimulator {

    fn state_vector() -> Vec<Complex<f32>> {
        // TODO: Calculate state vector
    }

    // Changes the state of the simulator according to its quantum circuit.
    fn perform_all_gates(self: mut Self) {
        for i: Instruction in self.circuit {
            // TODO: Apply gates, maybe crash on measurements for now?
        }
    }

    fn run(&mut self) -> Collapsed {
        self.perform_all_gates(); 
        let mut rng = rand::rng();
        let r = rng.random() // Random f64 in [0, 1)
        let last_probability_end = 0f64;
        for (bitstring, component) in self.calculate_state().enumerate() {
            let probability: f32 = component.pow(2).norm();
            let this_probability_end = last_probability_end + probability;
            if last_probability_end <= r && r < this_probability_end {
                return bitstring
            }
            last_probability_end = this_probability_end;
        }
        // If we did not collapse to any bitstring at all at this point,
        // something is very wrong!
        panic!("Tried to collapse state, but did not land on any bitstring!");
    }
}
