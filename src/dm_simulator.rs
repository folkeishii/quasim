use crate::{
    cart,
    circuit::Circuit,
    ext::expand_matrix_from_gate,
    instruction::Instruction,
};
use nalgebra::{Complex, DMatrix};
#[derive(Debug, Clone)]
pub struct DMSimulator {
    current_state: DMatrix<Complex<f64>>,
    circuit: Circuit,
    current_step: usize,
}

impl TryFrom<Circuit> for DMSimulator {
    type Error = DMSimulatorError;

    fn try_from(value: Circuit) -> Result<Self, Self::Error> {
        let circuit = value;
        let k = circuit.n_qubits();

        // Check for mid-cicuit measurement
        let mut encountered = false;
        for inst in circuit.instructions() {
            let _ = inst; // avoid warning for now
            let is_measurement = false; // matches!(inst, Instruction::Measurement(_));
            if is_measurement {
                encountered = true;
            } else if encountered {
                // There was a gate between measurements
                return Err(DMSimulatorError::MidCircuitMeasurement);
            }
        }

        // Initial state assumed to be |000..>
        // == |0><0| * |0><0| * |0><0| * ...
        let dim = 1 << k;
        let mut init_state = DMatrix::<Complex<f64>>::zeros(dim, dim);
        init_state[(0, 0)] = cart!(1.0);

        let sim = DMSimulator {
            current_state: init_state,
            circuit: circuit,
            current_step: 0,
        };

        Ok(sim)
    }
}

impl DMSimulator {
    fn next(&mut self) -> Option<&DMatrix<Complex<f64>>> {
        if self.current_step >= self.instruction_count() {
            return None;
        }
        match &self.circuit.instructions()[self.current_step] {
            Instruction::Gate(gate) => {
                let mat = expand_matrix_from_gate(&gate, self.circuit.n_qubits());
                self.current_state = mat.clone() * self.current_state.clone() * mat.adjoint(); // p1 == U * p0 * U'
            }
            _ => todo!(),
        }
        self.current_step += 1;
        Some(&self.current_state)
    }

    fn current_instruction(&self) -> Option<(usize, &Instruction)> {
        self.circuit
            .instructions()
            .get(self.current_step)
            .map(|inst| (self.current_step, inst))
    }

    fn current_state(&self) -> &DMatrix<Complex<f64>> {
        &self.current_state
    }

    fn prev(&mut self) -> Option<&DMatrix<Complex<f64>>> {
        if self.current_step <= 0 {
            return None;
        }
        self.current_step -= 1;
        match &self.circuit.instructions()[self.current_step] {
            Instruction::Gate(gate) => {
                let mut mat = expand_matrix_from_gate(&gate, self.circuit.n_qubits());
                mat.try_inverse_mut();
                self.current_state = mat.clone() * self.current_state.clone() * mat.adjoint(); // p0 == U^{-1} * p0 * (U^{-1})'
            }
            _ => todo!(), //Not possible without saving states before
                          //measurements.
        }
        Some(&self.current_state)
    }

    pub fn instruction_count(&self) -> usize {
        self.circuit.instructions().len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DMSimulatorError {
    #[error("Measurement mid-circuit")]
    MidCircuitMeasurement,
}

#[cfg(test)]
mod tests {
    use crate::{
        cart, circuit::Circuit, dm_simulator::DMSimulator, ext::reduced_state,
        simulator::BuildSimulator,
    };
    use nalgebra::{Complex, DMatrix, dmatrix};

    fn is_matrix_equal_to(m1: DMatrix<Complex<f64>>, m2: DMatrix<Complex<f64>>) -> bool {
        m1.iter()
            .zip(m2.iter())
            .all(|(a, b)| nalgebra::ComplexField::abs(a - b) < 0.001)
    }

    #[test]
    fn hadamard_cnot() {
        let mut sim = DMSimulator::build(Circuit::new(2).hadamard(0).cnot(0, 1)).unwrap();
        assert!(is_matrix_equal_to(
            sim.next().unwrap().clone(),
            dmatrix![
                cart!(0.5) ,cart!(0.5) ,cart!(0.0) ,cart!(0.0);
                cart!(0.5) ,cart!(0.5) ,cart!(0.0) ,cart!(0.0);
                cart!(0.0) ,cart!(0.0) ,cart!(0.0) ,cart!(0.0);
                cart!(0.0) ,cart!(0.0) ,cart!(0.0) ,cart!(0.0);

            ]
        ));
        assert!(is_matrix_equal_to(
            sim.next().unwrap().clone(),
            dmatrix![
                cart!(0.5) ,cart!(0.0) ,cart!(0.0) ,cart!(0.5);
                cart!(0.0) ,cart!(0.0) ,cart!(0.0) ,cart!(0.0);
                cart!(0.0) ,cart!(0.0) ,cart!(0.0) ,cart!(0.0);
                cart!(0.5) ,cart!(0.0) ,cart!(0.0) ,cart!(0.5);

            ]
        ));
    }

    #[test]
    fn reduced_state_hadamard_double_cnot() {
        let mut sim =
            DMSimulator::build(Circuit::new(3).hadamard(0).cnot(0, 1).cnot(0, 2)).unwrap();
        sim.next().unwrap();
        sim.next().unwrap();
        let rho = sim.next().unwrap();
        // From each qubit's perspecitve, its a 50/50 chance for either 0 or 1.
        let expected_individual_rho = dmatrix![
            cart!(0.5), cart!(0.0);
            cart!(0.0), cart!(0.5);
        ];
        let rho_0 = reduced_state(rho, &[0], 3);
        assert!(is_matrix_equal_to(expected_individual_rho.clone(), rho_0));
        let rho_1 = reduced_state(rho, &[1], 3);
        assert!(is_matrix_equal_to(expected_individual_rho.clone(), rho_1));
        let rho_2 = reduced_state(rho, &[2], 3);
        assert!(is_matrix_equal_to(expected_individual_rho, rho_2));
        let expected_pairs_rho = dmatrix![
            cart!(0.5), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0),cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0),cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0),cart!(0.0), cart!(0.0), cart!(0.5);
        ];
        let rho_01 = reduced_state(rho, &[0, 1], 3);
        assert!(is_matrix_equal_to(expected_pairs_rho.clone(), rho_01));
        let rho_12 = reduced_state(rho, &[1, 2], 3);
        assert!(is_matrix_equal_to(expected_pairs_rho.clone(), rho_12));
        let rho_02 = reduced_state(rho, &[0, 2], 3);
        assert!(is_matrix_equal_to(expected_pairs_rho, rho_02));

        let rho_reduce = reduced_state(rho, &[0, 1, 2], 3);
        assert!(is_matrix_equal_to(rho_reduce, rho.clone()));
    }
}
