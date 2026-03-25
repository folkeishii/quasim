use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit, pc::CircuitPc},
    expr_dsl::{Expr, Value},
    ext::{collapse_matrix, expand_matrix_from_gate, measure_and_observe_dm},
    gate::Gate,
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{DebuggableSimulator, StoredCircuitSimulator},
};
use nalgebra::{Complex, DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct DMSimulator {
    current_state: DMatrix<Complex<f64>>,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile<Value>,
    diagonal: DVector<Complex<f64>>,
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
impl DebuggableSimulator for DMSimulator {
    fn next(&mut self) -> Option<&DVector<Complex<f64>>> {
        let Some(inst) = self.circuit.instruction(self.pc()) else {
            return None;
        };

        match inst {
            Instruction::Gate(gate) => {
                self.apply_gate(gate);
                self.pc_mut().increment();
            }
            Instruction::MeasureBit(qbit, (reg, bit_pos)) => self.measure_bit(qbit, &reg, bit_pos),
            Instruction::MeasureAll(reg) => self.measure_all(&reg),
            Instruction::Jump(pc) => self.jump(pc),
            Instruction::JumpIf(expr, pc) => self.jump_if(&expr, pc),
            Instruction::Assign(expr, reg) => self.assign(&expr, &reg),
            Instruction::Call(_expr, _reg) => todo!(),
        }
        Some(&self.diagonal)
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>) {
        (self.pc(), self.circuit.instruction(self.pc()))
    }

    fn current_state(&self) -> &DVector<Complex<f64>> {
        todo!()
    }

    fn prev(&mut self) -> Option<&DVector<Complex<f64>>> {
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
            Instruction::Call(_, _) => todo!(),
        }
        Some(&self.diagonal)
    }

    fn double_ended(&self) -> bool {
        true
    }
}

impl DMSimulator {
    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        let (measurement, new_state) =
            measure_and_observe_dm(target, &self.current_state, self.n_qubits());

        let shifted_measurement = measurement << bit_pos;

        if let Value::Int(val) = self.registers[reg] {
            let val_cleared = (val as usize) & !shifted_measurement;
            self.registers[reg] = Value::Int((val_cleared | shifted_measurement) as i32)
        } else {
            self.registers[reg] = Value::Int(shifted_measurement as i32)
        }

        self.current_state = new_state;

        self.pc_mut().increment();
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement = collapse_matrix(&self.current_state);

        self.registers[reg] = Value::Int(measurement as i32);

        // Collapse whole density matrix
        self.current_state.fill(cart!(0.0));
        self.current_state[(measurement, measurement)] = cart!(1.0);

        self.pc_mut().increment();
    }

    fn jump(&mut self, label_pc: usize) {
        self.pc_mut().jump(label_pc);
    }

    fn jump_if(&mut self, expr: &Expr, label_pc: usize) {
        match expr.eval(&self.registers) {
            Ok(Value::Bool(true)) => self.jump(label_pc),
            Ok(Value::Bool(false)) => self.pc_mut().increment(),
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
        self.pc_mut().increment();
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

        let registers = RegisterFile::from(circuit.registers());

        DMSimulator {
            diagonal: init_state.diagonal(),
            current_state: init_state,
            circuit: circuit,
            pc: Default::default(),
            registers: registers,
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
    use crate::ext::{equal_to_matrix_c, reduced_state};
    use crate::{
        cart,
        circuit::Circuit,
        dm_simulator::DMSimulator,
        expr_dsl::{Value, expr_helpers::r},
        simulator::DebuggableSimulator,
    };
    use nalgebra::{Complex, DMatrix, dmatrix};

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
        assert!(equal_to_matrix_c(&sim.current_state, &expected_mat, 0.001));
    }

    #[test]
    fn hchch_test() {
        let mut sim = DMSimulator::init(Circuit::new(3).h(0).ch(&[0], 1).ch(&[1], 2).into());
        sim.next();
        sim.next();
        sim.next();
        let rho_0 = dmatrix![
            cart!(0.5), cart!(0.353553);
            cart!(0.353553), cart!(0.5)
        ];
        let rho_1 = dmatrix![
            cart!(0.75), cart!(0.176777);
            cart!(0.176777), cart!(0.25)
        ];
        let rho_2 = dmatrix![
            cart!(0.875), cart!(0.125);
            cart!(0.125), cart!(0.125)
        ];
        let rho_01 = dmatrix![
            cart!(0.5)     , cart!(0.353553), cart!(0.0), cart!(0.25);
            cart!(0.353553), cart!(0.25)    , cart!(0.0), cart!(0.176777);
            cart!(0.0)     , cart!(0.0)     , cart!(0.0), cart!(0.0);
            cart!(0.25)    , cart!(0.176777), cart!(0.0), cart!(0.25);
        ];
        let rho_12 = dmatrix![
            cart!(0.75)    , cart!(0.176777), cart!(0.0), cart!(0.176777);
            cart!(0.176777), cart!(0.125)    , cart!(0.0), cart!(0.125);
            cart!(0.0)     , cart!(0.0)     , cart!(0.0), cart!(0.0);
            cart!(0.176777), cart!(0.125)   , cart!(0.0), cart!(0.125);
        ];
        let rho_012 = dmatrix![
            cart!(0.5)     , cart!(0.353553), cart!(0.0), cart!(0.25)    , cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.25);
            cart!(0.353553), cart!(0.25)    , cart!(0.0), cart!(0.176777), cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.176777);
            cart!(0.0)     , cart!(0.0)     , cart!(0.0), cart!(0.0)     , cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.25)    , cart!(0.176777), cart!(0.0), cart!(0.125)   , cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.125);
            cart!(0.0)     , cart!(0.0)     , cart!(0.0), cart!(0.0)     , cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0)     , cart!(0.0)     , cart!(0.0), cart!(0.0)     , cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.0)     , cart!(0.0)     , cart!(0.0), cart!(0.0)     , cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.0);
            cart!(0.25)    , cart!(0.176777), cart!(0.0), cart!(0.125)   , cart!(0.0), cart!(0.0), cart!(0.0), cart!(0.125);
        ];
        let rho = sim.current_state.clone();
        assert!(equal_to_matrix_c(
            &rho_0,
            &reduced_state(&rho, &[0], 3),
            0.001
        ));
        assert!(equal_to_matrix_c(
            &rho_1,
            &reduced_state(&rho, &[1], 3),
            0.001
        ));
        assert!(equal_to_matrix_c(
            &rho_2,
            &reduced_state(&rho, &[2], 3),
            0.001
        ));
        assert!(equal_to_matrix_c(
            &rho_01,
            &reduced_state(&rho, &[0, 1], 3),
            0.001
        ));
        assert!(equal_to_matrix_c(
            &rho_12,
            &reduced_state(&rho, &[1, 2], 3),
            0.001
        ));
        assert!(equal_to_matrix_c(
            &rho_012,
            &reduced_state(&rho, &[0, 1, 2], 3),
            0.001
        ));
        assert!(equal_to_matrix_c(&rho, &rho_012, 0.001));
        sim.prev();
        sim.prev();
        sim.prev();

        let mut rho_init = DMatrix::<Complex<f64>>::zeros(8, 8);
        rho_init[(0, 0)] = cart!(1.0);
        assert!(equal_to_matrix_c(&rho_init, &sim.current_state, 0.001));
    }

    #[test]
    fn hybrid_test() {
        let circuit = Circuit::new(4)
            .new_reg("r0")
            .new_reg("r1")
            .new_reg("r2")
            .new_reg("r3")
            // Init random state
            .h(0)
            .h(1)
            .h(2)
            .h(3)
            .measure_bit(0, ("r0", 0))
            .measure_bit(1, ("r1", 0))
            .measure_bit(2, ("r2", 0))
            .measure_bit(3, ("r3", 0))
            .apply_if(r("r0").eq(1))
            .x(0)
            .apply_if(r("r1").eq(1))
            .x(1)
            .apply_if(r("r2").eq(1))
            .x(2)
            .apply_if(r("r3").eq(1))
            .x(3);

        let mut sim = DMSimulator::init(circuit);
        while let Some(_) = sim.next() {}

        let mut expected = DMatrix::<Complex<f64>>::zeros(16, 16);
        expected[(0, 0)] = cart!(1.0);

        assert!(equal_to_matrix_c(&sim.current_state, &expected, 0.001));
    }

    #[test]
    fn register_test() {
        let circuit = Circuit::new(2).new_reg("r0").x(1).measure_bit(1, ("r0", 0));

        let mut sim = DMSimulator::init(circuit);
        while let Some(_) = sim.next() {}

        assert_eq!(sim.registers["r0"], Value::Int(1));
    }
}
