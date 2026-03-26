use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit, pc::CircuitPc},
    expr_dsl::{Expr, Value},
    ext::{
        collapse_matrix, eval_tensor_product, expand_matrix_from_gate, measure_and_observe_dm,
        swap_matrix,
    },
    gate::Gate,
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{DebuggableSimulator, StoredCircuitSimulator},
};
use nalgebra::{Complex, DMatrix, DVector, dmatrix, dvector};

#[derive(Debug, Clone)]
pub struct QSys {
    density: DMatrix<Complex<f64>>,
    qubits: Vec<usize>,
}

impl std::fmt::Display for QSys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Qubits: {:?}\n Density: {}", self.qubits, self.density)
    }
}

impl QSys {
    fn add_system(&self, rhs: &QSys) -> QSys {
        let mut qubits = self.qubits.clone();
        qubits.extend(rhs.qubits.clone());
        QSys {
            density: eval_tensor_product(vec![self.density.clone(), rhs.density.clone()]),
            qubits: qubits,
        }
    }

    fn to_local_index(&self, global_index: usize) -> usize {
        let Some(local_index) = self.qubits.iter().position(|&q| q == global_index) else {
            panic!("Wtf?")
        };
        local_index
    }
}

#[derive(Debug, Clone)]
pub struct DMSimulator {
    qsystems: Vec<QSys>,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile<Value>,
    dummy_state: DVector<Complex<f64>>, //TODO: when next/prev returns bool, remove this
}

impl DMSimulator {
    fn init(circuit: Circuit<HybridCircuit>) -> Self {
        // Initial state assumed to be |000..>
        // == |0><0| * |0><0| * |0><0| * ...

        let mut init_sys = vec![];

        for i in 0..circuit.n_qubits() {
            init_sys.push(QSys {
                density: dmatrix![cart!(1.0), cart!(0.0); cart!(0.0), cart!(0.0)], // |0><0|
                qubits: vec![i],
            });
        }

        let registers = RegisterFile::from(circuit.registers());

        DMSimulator {
            qsystems: init_sys,
            circuit: circuit,
            pc: Default::default(),
            registers: registers,
            dummy_state: dvector![], //TODO: when next/prev returns bool, remove this
        }
    }

    fn find_system_of_qubit(&self, qubit: usize) -> usize {
        let Some(qsys_idx) = self.qsystems.iter().position(|s| s.qubits.contains(&qubit)) else {
            panic!("Wtf?")
        };
        qsys_idx
    }

    fn apply_gate(&mut self, gate: Gate) {
        // Om > 1 qubit -> merge systems
        // Annars, applya som vanligt
        let controls = gate.get_controls();
        let targets = gate.get_targets();
        let arity = controls.len() + targets.len();

        if arity == 1 {
            //println!("APPLY: single qubit gate");
            // Single qubit gate
            let global_target = targets[0];
            let qsys_idx = self.find_system_of_qubit(global_target);
            let qsys = &self.qsystems[qsys_idx];
            let local_target = qsys.to_local_index(global_target);
            let local_gate = Gate::new(gate.get_type(), &[], &[local_target]).unwrap();
            let n_qubits = qsys.qubits.len();
            let mat = expand_matrix_from_gate(&local_gate, n_qubits);
            let mat_adj = mat.adjoint();
            let density = qsys.density.clone();
            self.qsystems[qsys_idx].density = mat * density * mat_adj;
            return;
        }
        // Multi qubit gate

        //println!("APPLY: multi qubit gate");
        let mut gate_qubits = controls.clone();
        gate_qubits.extend(&targets);

        let mut gate_qsystems = vec![];
        for qubit in gate_qubits {
            let qsys_idx = self.find_system_of_qubit(qubit);
            if !gate_qsystems.contains(&qsys_idx) {
                gate_qsystems.push(qsys_idx);
            }
        }

        if gate_qsystems.len() > 1 {
            //println!("APPLY: multi system gate");
            // Multi system gate
            // -> Combine systems
            ////
            //for sys in gate_qsystems.clone() {
            //    println!("SYS: {}", self.qsystems[sys]);
            //}
            ////
            let mut tot_qsys = gate_qsystems
                .iter()
                .map(|&qsys_idx| self.qsystems[qsys_idx].clone())
                .fold(
                    QSys {
                        density: dmatrix![cart!(1.0)],
                        qubits: vec![],
                    },
                    |acc, qsys| acc.add_system(&qsys),
                );
            //println!("SYS_TOT: {}", tot_qsys);

            let local_targets: Vec<usize> = targets
                .iter()
                .map(|&t| tot_qsys.to_local_index(t))
                .collect();

            let local_controls: Vec<usize> = controls
                .iter()
                .map(|&t| tot_qsys.to_local_index(t))
                .collect();

            let local_n_qubits = tot_qsys.qubits.len();
            let local_gate = Gate::new(gate.get_type(), &local_controls, &local_targets).unwrap();
            let mat = expand_matrix_from_gate(&local_gate, local_n_qubits);
            let mat_adj = mat.adjoint();
            tot_qsys.density = mat * tot_qsys.density * mat_adj;

            // remove systems that were combined
            self.qsystems = self
                .qsystems
                .iter()
                .enumerate()
                .filter(|(i, _)| !gate_qsystems.contains(i))
                .map(|(_, q)| q.clone())
                .collect();
            // add combined system
            self.qsystems.push(tot_qsys);
            return;
        }
        //println!("APPLY: single system gate");

        // Multi qubit, single system gate
        let mut qsys = self.qsystems[gate_qsystems[0]].clone();
        let local_targets: Vec<usize> = targets.iter().map(|&t| qsys.to_local_index(t)).collect();

        let local_controls: Vec<usize> = controls.iter().map(|&t| qsys.to_local_index(t)).collect();

        let local_n_qubits = qsys.qubits.len();
        let local_gate = Gate::new(gate.get_type(), &local_controls, &local_targets).unwrap();
        let mat = expand_matrix_from_gate(&local_gate, local_n_qubits);
        let mat_adj = mat.adjoint();
        qsys.density = mat * qsys.density * mat_adj;
        self.qsystems[gate_qsystems[0]] = qsys;
    }

    fn density(&self) -> DMatrix<Complex<f64>> {
        let mut tot_qsys = (0..self.qsystems.len())
            .into_iter()
            .map(|qsys_idx| self.qsystems[qsys_idx].clone())
            .fold(
                QSys {
                    density: dmatrix![cart!(1.0)],
                    qubits: vec![],
                },
                |acc, qsys| acc.add_system(&qsys),
            );
        // insertion sort with swap
        //i ← 1
        //while i < length(A)
        //    j ← i
        //    while j > 0 and A[j-1] > A[j]
        //        swap A[j] and A[j-1]
        //        j ← j - 1
        //    end while
        //    i ← i + 1
        //end while
        let n_qubits = tot_qsys.qubits.len();
        let mut i = 1;
        while i < n_qubits {
            let mut j = i;
            while j > 0 && tot_qsys.qubits[j - 1] > tot_qsys.qubits[j] {
                tot_qsys.qubits.swap(j, j - 1);

                let mat = swap_matrix(&[], j, j - 1, n_qubits);
                let mat_adj = mat.adjoint();
                tot_qsys.density = mat * tot_qsys.density * mat_adj;

                j -= 1;
            }
            i += 1;
        }
        tot_qsys.density
    }

    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        todo!();
        self.pc_mut().increment();
    }

    fn measure_all(&mut self, reg: &str) {
        todo!();
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
            Instruction::Call(name, lsq, ctrl) => self.pc_mut().jump_and_link(name, lsq, ctrl),
        }
        Some(&self.dummy_state) //TODO: when next/prev returns bool, remove this
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>) {
        (self.pc(), self.circuit.instruction(self.pc()))
    }

    fn current_state(&self) -> &DVector<Complex<f64>> {
        &self.dummy_state //TODO: when next/prev returns bool, remove this
    }

    fn double_ended(&self) -> bool {
        false
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
        println!("Sim initalized");
        sim.next();
        println!("Sim stepped once");
        sim.next();
        println!("Sim stepped twice");
        let expected_mat = dmatrix![
            cart!(0.5)     , cart!(0.353553), cart!(0.0), cart!(0.353553);
            cart!(0.353553), cart!(0.25)    , cart!(0.0), cart!(0.25);
            cart!(0.0)     , cart!(0.0)     , cart!(0.0), cart!(0.0);
            cart!(0.353553), cart!(0.25)    , cart!(0.0), cart!(0.25);
        ];
        assert!(equal_to_matrix_c(&sim.density(), &expected_mat, 0.001));
    }

    fn print_systems(sim: &DMSimulator) {
        for sys in sim.qsystems.clone() {
            println!("{}", sys);
        }
    }

    #[test]
    fn interleaved_ch_test() {
        let mut sim = DMSimulator::init(
            Circuit::new(4)
                .h(0)
                .ch(&[0], 2)
                .h(1)
                .ch(&[1], 3)
                .ch(&[2], 1)
                .ch(&[0], 3)
                .into(),
        );
        sim.next();
        sim.next();
        sim.next();
        sim.next();
        sim.next();
        sim.next();
        let mut expected = DMatrix::<Complex<f64>>::zeros(16, 16);

        expected[(0, 0)] = cart!(0.2500000298023224);
        expected[(0, 1)] = cart!(0.125);
        expected[(0, 2)] = cart!(0.1767767071723938);
        expected[(0, 3)] = cart!(0.1767767071723938);
        expected[(0, 5)] = cart!(0.2133883386850357);
        expected[(0, 7)] = cart!(-0.0366116501390934);
        expected[(0, 9)] = cart!(0.125);
        expected[(0, 10)] = cart!(0.1767767071723938);
        expected[(0, 13)] = cart!(0.0883883386850357);
        expected[(0, 15)] = cart!(0.0883883535861969);

        expected[(1, 0)] = cart!(0.125);
        expected[(1, 1)] = cart!(0.0625);
        expected[(1, 2)] = cart!(0.0883883535861969);
        expected[(1, 3)] = cart!(0.0883883535861969);
        expected[(1, 5)] = cart!(0.10669416189193726);
        expected[(1, 7)] = cart!(-0.0183058250695467);
        expected[(1, 9)] = cart!(0.0625);
        expected[(1, 10)] = cart!(0.0883883535861969);
        expected[(1, 13)] = cart!(0.04419416934251785);
        expected[(1, 15)] = cart!(0.04419417679309845);

        expected[(2, 0)] = cart!(0.1767767071723938);
        expected[(2, 1)] = cart!(0.0883883535861969);
        expected[(2, 2)] = cart!(0.125);
        expected[(2, 3)] = cart!(0.125);
        expected[(2, 5)] = cart!(0.1508883386850357);
        expected[(2, 7)] = cart!(-0.025888346135616302);
        expected[(2, 9)] = cart!(0.0883883535861969);
        expected[(2, 10)] = cart!(0.125);
        expected[(2, 13)] = cart!(0.0624999925494194);
        expected[(2, 15)] = cart!(0.0625);

        expected[(3, 0)] = cart!(0.1767767071723938);
        expected[(3, 1)] = cart!(0.0883883535861969);
        expected[(3, 2)] = cart!(0.125);
        expected[(3, 3)] = cart!(0.125);
        expected[(3, 5)] = cart!(0.1508883386850357);
        expected[(3, 7)] = cart!(-0.025888346135616302);
        expected[(3, 9)] = cart!(0.0883883535861969);
        expected[(3, 10)] = cart!(0.125);
        expected[(3, 13)] = cart!(0.0624999925494194);
        expected[(3, 15)] = cart!(0.0625);

        expected[(5, 0)] = cart!(0.2133883386850357);
        expected[(5, 1)] = cart!(0.10669416189193726);
        expected[(5, 2)] = cart!(0.1508883386850357);
        expected[(5, 3)] = cart!(0.1508883386850357);
        expected[(5, 5)] = cart!(0.1821383237838745);
        expected[(5, 7)] = cart!(-0.0312499962747097);
        expected[(5, 9)] = cart!(0.10669416189193726);
        expected[(5, 10)] = cart!(0.1508883386850357);
        expected[(5, 13)] = cart!(0.07544415444135666);
        expected[(5, 15)] = cart!(0.07544416934251785);

        expected[(7, 0)] = cart!(-0.0366116501390934);
        expected[(7, 1)] = cart!(-0.0183058250695467);
        expected[(7, 2)] = cart!(-0.025888346135616302);
        expected[(7, 3)] = cart!(-0.025888346135616302);
        expected[(7, 5)] = cart!(-0.0312499962747097);
        expected[(7, 7)] = cart!(0.005361652001738548);
        expected[(7, 9)] = cart!(-0.0183058250695467);
        expected[(7, 10)] = cart!(-0.025888346135616302);
        expected[(7, 13)] = cart!(-0.012944171205163002);
        expected[(7, 15)] = cart!(-0.012944173067808151);

        expected[(9, 0)] = cart!(0.125);
        expected[(9, 1)] = cart!(0.0625);
        expected[(9, 2)] = cart!(0.0883883535861969);
        expected[(9, 3)] = cart!(0.0883883535861969);
        expected[(9, 5)] = cart!(0.10669416189193726);
        expected[(9, 7)] = cart!(-0.0183058250695467);
        expected[(9, 9)] = cart!(0.0625);
        expected[(9, 10)] = cart!(0.0883883535861969);
        expected[(9, 13)] = cart!(0.04419416934251785);
        expected[(9, 15)] = cart!(0.04419417679309845);

        expected[(10, 0)] = cart!(0.1767767071723938);
        expected[(10, 1)] = cart!(0.0883883535861969);
        expected[(10, 2)] = cart!(0.125);
        expected[(10, 3)] = cart!(0.125);
        expected[(10, 5)] = cart!(0.1508883386850357);
        expected[(10, 7)] = cart!(-0.025888346135616302);
        expected[(10, 9)] = cart!(0.0883883535861969);
        expected[(10, 10)] = cart!(0.125);
        expected[(10, 13)] = cart!(0.0624999925494194);
        expected[(10, 15)] = cart!(0.0625);

        expected[(13, 0)] = cart!(0.0883883386850357);
        expected[(13, 1)] = cart!(0.04419416934251785);
        expected[(13, 2)] = cart!(0.0624999925494194);
        expected[(13, 3)] = cart!(0.0624999925494194);
        expected[(13, 5)] = cart!(0.07544415444135666);
        expected[(13, 7)] = cart!(-0.012944171205163002);
        expected[(13, 9)] = cart!(0.04419416934251785);
        expected[(13, 10)] = cart!(0.0624999925494194);
        expected[(13, 13)] = cart!(0.031249990686774254);
        expected[(13, 15)] = cart!(0.0312499962747097);

        expected[(15, 0)] = cart!(0.0883883535861969);
        expected[(15, 1)] = cart!(0.04419417679309845);
        expected[(15, 2)] = cart!(0.0625);
        expected[(15, 3)] = cart!(0.0625);
        expected[(15, 5)] = cart!(0.07544416934251785);
        expected[(15, 7)] = cart!(-0.012944173067808151);
        expected[(15, 9)] = cart!(0.04419417679309845);
        expected[(15, 10)] = cart!(0.0625);
        expected[(15, 13)] = cart!(0.0312499962747097);
        expected[(15, 15)] = cart!(0.03125);

        assert!(equal_to_matrix_c(&sim.density(), &expected, 0.001));
    }
}
