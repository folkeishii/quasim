use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit, pc::CircuitPc},
    ext::{expand_matrix_from_gate, measure_and_observe_dm},
    gate::Gate,
    instruction::Instruction,
    simulator::StoredCircuitSimulator,
};
use nalgebra::{Complex, DMatrix};

#[derive(Debug, Clone)]
pub struct DMSimulator {
    current_state: DMatrix<Complex<f64>>,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
}

impl TryFrom<Circuit<PureCircuit>> for DMSimulator {
    type Error = DMSimulatorError;

    fn try_from(value: Circuit<PureCircuit>) -> Result<Self, Self::Error> {
        Self::try_from(Circuit::<HybridCircuit>::from(value.into()))
    }
}

impl TryFrom<Circuit<HybridCircuit>> for DMSimulator {
    type Error = DMSimulatorError;

    fn try_from(value: Circuit<HybridCircuit>) -> Result<Self, Self::Error> {
        let circuit = value;

        // Check for mid-cicuit measurement
        let mut encountered = false;
        for inst in circuit.instructions() {
            let is_measurement = matches!(inst, Instruction::MeasureBit(_, _))
                || matches!(inst, Instruction::MeasureAll(_));
            if is_measurement {
                encountered = true;
            } else if encountered {
                // There was a gate between measurements
                return Err(DMSimulatorError::MidCircuitMeasurement);
            }
        }

        let sim = Self::init(circuit);

        Ok(sim)
    }
}

impl DMSimulator {
    fn next(&mut self) -> Option<&DMatrix<Complex<f64>>> {
        let Some(inst) = self.circuit.instruction(self.pc()) else {
            return None;
        };

        match inst {
            Instruction::Gate(gate) => {
                self.apply_gate(gate);
                self.pc_mut().increment();
            }
            Instruction::MeasureBit(qbit, _) => {
                let (_res, new_state) =
                    measure_and_observe_dm(qbit, &self.current_state, self.n_qubits());
                self.current_state = new_state;
                self.pc_mut().increment();
            }
            Instruction::MeasureAll(_) => todo!(),
            Instruction::Jump(_) => todo!(),
            Instruction::JumpIf(_, _) => todo!(),
            Instruction::Assign(_, _) => todo!(),
        }
        Some(&self.current_state)
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>) {
        (self.pc(), self.circuit.instruction(self.pc()))
    }

    fn current_state(&self) -> &DMatrix<Complex<f64>> {
        &self.current_state
    }

    fn prev(&mut self) -> Option<&DMatrix<Complex<f64>>> {
        if !self.pc_mut().decrement() {
            return None;
        }

        let Some(inst) = self.circuit.instruction(self.pc()) else {
            // Should not happen
            return None;
        };

        match inst {
            Instruction::Gate(gate) => self.apply_gate_inv(gate),
            Instruction::MeasureBit(_, _) => todo!(),
            Instruction::MeasureAll(_) => todo!(),
            Instruction::Jump(_) => todo!(),
            Instruction::JumpIf(_, _) => todo!(),
            Instruction::Assign(_, _) => todo!(),
        }
        Some(&self.current_state)
    }

    fn double_ended(&self) -> bool {
        true
    }

    fn pc(&self) -> &CircuitPc {
        &self.pc
    }

    fn pc_mut(&mut self) -> &mut CircuitPc {
        &mut self.pc
    }

    fn apply_gate(&mut self, gate: Gate) {
        let mat = expand_matrix_from_gate(&gate, self.circuit.n_qubits());
        let adj = mat.adjoint();
        self.current_state = mat * self.current_state.clone() * adj;
    }

    fn apply_gate_inv(&mut self, gate: Gate) {
        let adj_inv = expand_matrix_from_gate(&gate, self.circuit.n_qubits());
        let mat_inv = adj_inv.adjoint();
        self.current_state = mat_inv * self.current_state.clone() * adj_inv;
    }

    fn init(circuit: Circuit<HybridCircuit>) -> Self {
        // Initial state assumed to be |000..>
        // == |0><0| * |0><0| * |0><0| * ...
        let dim = 1 << circuit.n_qubits();
        let mut init_state = DMatrix::<Complex<f64>>::zeros(dim, dim);
        init_state[(0, 0)] = cart!(1.0);

        DMSimulator {
            current_state: init_state,
            circuit: circuit,
            pc: Default::default(),
        }
    }
}

impl StoredCircuitSimulator for DMSimulator {
    type B = HybridCircuit;
    fn circuit(&self) -> &Circuit<HybridCircuit> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<HybridCircuit> {
        &mut self.circuit
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum DMSimulatorError {
    #[error("Measurement mid-circuit")]
    MidCircuitMeasurement,
}

#[cfg(test)]
mod tests {
    use crate::ext::equal_to_matrix_c;
    use crate::{
        cart,
        circuit::Circuit,
        dm_simulator::DMSimulator,
    };
    use nalgebra::dmatrix;

    #[test]
    fn hch_test() {
        let mut sim = DMSimulator::init(Circuit::new(2).h(0).ch(&[0], 1).into());
        sim.next();
        sim.next();
        let expected_mat = dmatrix![
            cart!(0.5)     , cart!(0.353553), cart!(0.0), cart!(0.353553);
            cart!(0.353553), cart!(0.25)    , cart!(0.0), cart!(0.25);
            cart!(0.0)     , cart!(0.0)     , cart!(0.0), cart!(0.0);
            cart!(0.353553), cart!(0.25)    , cart!(0.0), cart!(0.25);
        ];
        assert!(equal_to_matrix_c(&sim.current_state,&expected_mat,0.001));
    }
}
