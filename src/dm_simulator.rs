use crate::{
    cart,
    circuit::Circuit,
    ext::{eval_tensor_product, expand_matrix_from_gate, get_gate_matrix, identity_tensor_factors},
    gate::Gate,
    instruction::Instruction,
};
use nalgebra::{Complex, DMatrix, dmatrix};

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
    fn reduced_state(
        density: &DMatrix<Complex<f64>>,
        targets: &[usize],
        n_qubits: usize,
    ) -> DMatrix<Complex<f64>> {
        let bra = [
            dmatrix![cart!(1.0), cart!(0.0)], // <0|
            dmatrix![cart!(0.0), cart!(1.0)], // <0|
        ];

        let dim = 1 << targets.len();
        let mut sum = DMatrix::<Complex<f64>>::zeros(dim, dim);

        let non_targets = (0..n_qubits)
            .filter(|idx| !targets.contains(idx))
            .collect::<Vec<usize>>();
        let n_terms = 1 << (n_qubits - targets.len());
        for i in 0..n_terms {
            /* Example, targets = [2], n_qubits = 3:
             *
             * p_2 = (<0| * <0| * I)p(|0> * |0> * I) +
             *     + (<0| * <1| * I)p(|0> * |1> * I) +
             *     + (<1| * <0| * I)p(|1> * |0> * I) +
             *     + (<1| * <1| * I)p(|1> * |1> * I) +
             * */
            let mut left_of_density_prod = identity_tensor_factors(n_qubits);
            let mut j: usize = 0;
            for non_target in non_targets.iter() {
                left_of_density_prod[*non_target] = bra[(i >> j) & 1].clone();
                j += 1;
            }
            let left_of_density = eval_tensor_product(left_of_density_prod);
            let right_of_density = left_of_density.adjoint();
            sum += left_of_density * density * right_of_density;
        }
        sum
    }

    fn next(&mut self) -> Option<&DMatrix<Complex<f64>>> {
        if self.current_step >= self.instruction_count() {
            return None;
        }
        match &self.circuit.instructions()[self.current_step] {
            Instruction::Gate(gate) => {
                let mat = expand_matrix_from_gate(&gate, self.circuit.n_qubits());
                self.current_state = mat.clone() * self.current_state.clone() * mat.adjoint(); // p1 == U * p0 * U'
            }
            Instruction::Measurement(_qbits) => todo!(),
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
            Instruction::Measurement(_) => todo!(), //Not possible without saving states before
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
    use crate::ext::collapse;
    use crate::{
        cart,
        circuit::Circuit,
        dm_simulator::DMSimulator,
        ext::get_gate_matrix,
        gate::{Gate, GateType},
        simulator::{BuildSimulator, DebuggableSimulator, DoubleEndedSimulator},
    };
    use nalgebra::{Complex, DMatrix, DVector, dmatrix, dvector};
    use std::f64::consts::FRAC_1_SQRT_2;

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
        sim.next().expect("1");
        sim.next().expect("2");
        let rho = sim.next().expect("3");
        //println!("{}",rho);
        let rho_0 = DMSimulator::reduced_state(rho, &[0], 3);
        println!("{}", rho_0);
    }
}
