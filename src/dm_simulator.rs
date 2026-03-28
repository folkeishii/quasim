use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit, pc::CircuitPc},
    expr_dsl::{Expr, Value},
    ext::{
        collapse_probs, eval_tensor_product, expand_matrix_from_gate, measure_and_observe_dm,
        reduced_state, swap_matrix,
    },
    gate::{Gate, GateType},
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{DebuggableSimulator, StoredCircuitSimulator},
};
use nalgebra::{Complex, DMatrix, DVector, dmatrix, dvector};

/// A system of potentially entangled qubits.
#[derive(Debug, Clone)]
struct QSys {
    density: DMatrix<Complex<f64>>,
    qubits: Vec<usize>,
}

impl std::fmt::Display for QSys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Qubits: {:?}\n Density: {}", self.qubits, self.density)
    }
}

impl QSys {
    /// Concatinates qubit lists and "tensors" density matricies.
    fn add_system(&self, rhs: &Self) -> Self {
        let mut qubits = self.qubits.clone();
        qubits.extend(rhs.qubits.clone());
        Self {
            density: eval_tensor_product(vec![self.density.clone(), rhs.density.clone()]),
            qubits: qubits,
        }
    }

    fn to_local_index(&self, global_index: usize) -> usize {
        let Some(local_index) = self.qubits.iter().position(|&q| q == global_index) else {
            panic!("Qubit is not member of system!")
        };
        local_index
    }

    /// Insersion sort by qubit index and swaps on density accordingly.
    /// swap(target1: usize, target2: usize, n_qubits: usize, density: &DMatrix<Complex<f64>>)
    fn sort_with(
        &mut self,
        swap: fn(usize, usize, usize, &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>>,
    ) {
        let n_qubits = self.qubits.len();
        let mut i = 1;
        while i < n_qubits {
            let mut j = i;
            while j > 0 && self.qubits[j - 1] > self.qubits[j] {
                // Sort the list of qubit indecies.
                self.qubits.swap(j, j - 1);

                // Sort the density matrix.
                self.density = swap(j, j - 1, n_qubits, &self.density);

                j -= 1;
            }
            i += 1;
        }
    }
}

fn qsystem_product(qsystems: &[QSys]) -> QSys {
    qsystems.into_iter().fold(
        QSys {
            density: dmatrix![cart!(1.0)],
            qubits: vec![],
        },
        |acc, qsys| acc.add_system(&qsys),
    )
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
        // No entanglement -> one system for each qubit.

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
            panic!("Qubit is not member of any system!")
        };
        qsys_idx
    }

    fn apply_gate(&mut self, gate: Gate) {
        /* Overview:
         *
         *      - if gate acts on a single system,
         *      then apply gate to that system only.
         *
         *      - if gate acts on multiple systems,
         *      then combine those systems and then
         *      apply gate to the total system.
         *
         * */

        self.pc_mut().increment();

        let controls = gate.get_controls();
        let targets = gate.get_targets();

        // Qubits that gate acts on.
        let mut gate_qubits = controls.clone();
        gate_qubits.extend(&targets);

        // Systems that gate acts on.
        let mut gate_qsystems = vec![];
        for qubit in gate_qubits {
            let qsys_idx = self.find_system_of_qubit(qubit);
            if !gate_qsystems.contains(&qsys_idx) {
                gate_qsystems.push(qsys_idx);
            }
        }

        let mut qsys = self.qsystems[gate_qsystems[0]].clone();

        let gate_acts_on_several_systems = gate_qsystems.len() > 1;
        let is_swap_gate = gate.get_type() == GateType::SWAP && controls.is_empty();

        if gate_acts_on_several_systems && is_swap_gate {
            // Regular swaps do not result in entanglement.
            // Only change global qubit index.
            let qsys1_local_target = qsys.to_local_index(targets[0]);
            let mut qsys2 = self.qsystems[gate_qsystems[1]].clone();
            let qsys2_local_target = qsys2.to_local_index(targets[1]);

            // Swap.
            let temp = qsys.qubits[qsys1_local_target];
            qsys.qubits[qsys1_local_target] = qsys2.qubits[qsys2_local_target];
            qsys2.qubits[qsys2_local_target] = temp;

            self.qsystems[gate_qsystems[0]] = qsys;
            self.qsystems[gate_qsystems[1]] = qsys2;
            return;
        } else if gate_acts_on_several_systems {
            // Combine all systems acted on.
            qsys = gate_qsystems
                .iter()
                .skip(1)
                .map(|&qs_idx| self.qsystems[qs_idx].clone())
                .fold(qsys, |acc, qs| acc.add_system(&qs));
        }

        // Translate global qubit indexing to the system's local indexing.
        let local_targets: Vec<usize> = targets.iter().map(|&t| qsys.to_local_index(t)).collect();

        if is_swap_gate {
            // Swap qubit indecies, no need for matrix multiplication.
            qsys.qubits.swap(local_targets[0], local_targets[1]);
            self.qsystems[gate_qsystems[0]] = qsys;
            return;
        }

        // Continue to translate global qubit indexing to the system's local indexing.
        let local_controls: Vec<usize> = controls.iter().map(|&t| qsys.to_local_index(t)).collect();
        let local_n_qubits = qsys.qubits.len();
        let local_gate = Gate::new(gate.get_type(), &local_controls, &local_targets).unwrap();

        // Apply the gate to the system's density matrix using: p` == UpU'
        let mat = expand_matrix_from_gate(&local_gate, local_n_qubits);
        let mat_adj = mat.adjoint();
        qsys.density = mat * qsys.density * mat_adj;

        if gate_acts_on_several_systems {
            // Remove systems that were combined.
            self.qsystems = self
                .qsystems
                .iter()
                .enumerate()
                .filter(|(i, _)| !gate_qsystems.contains(i))
                .map(|(_, q)| q.clone())
                .collect();

            // Add combined system.
            self.qsystems.push(qsys);
            return;
        }
        // Update system acted on.
        self.qsystems[gate_qsystems[0]] = qsys;
    }
    /// The density matrix of the full system.
    fn density(&self) -> DMatrix<Complex<f64>> {
        // Combine all sub-systems into a single one.
        let mut tot_qsys = qsystem_product(&self.qsystems);

        tot_qsys.sort_with(|t1, t2, n, p| {
            let mat = swap_matrix(&[], t1, t2, n);
            let mat_adj = mat.clone();
            mat * p * mat_adj
        });

        tot_qsys.density
    }

    /// Just the diagonal of the full density matrix.
    fn probabilities(&self) -> Vec<f64> {
        // Ok to combine diagonals in the same way
        // as density matricies.
        let qsys_diags = self
            .qsystems
            .iter()
            .map(|qsys| QSys {
                density: DMatrix::<Complex<f64>>::from_columns(&[qsys.density.diagonal()]),
                qubits: qsys.qubits.clone(),
            })
            .collect::<Vec<QSys>>();

        let mut tot_qsys_diag = qsystem_product(&qsys_diags);

        tot_qsys_diag.sort_with(|t1, t2, n, v| {
            let mat = swap_matrix(&[], t1, t2, n);
            mat * v
        });

        tot_qsys_diag
            .density
            .iter()
            .map(|c| c.re)
            .collect::<Vec<f64>>()
    }

    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        /* Idea:
         *      - Once a qubit is measured and observed
         *      it can not be entangled with any other
         *      system.
         *      --> Measurements "splits" the measured system.
         *
         *      ex: 3 qubits A,B,C that might be entangled, A measured to |0>
         *
         *                      p` == |0><0| * Tr_BC(p)
         * */

        let target_qsystem = self.find_system_of_qubit(target);
        let mut qsys = self.qsystems[target_qsystem].clone();

        // Translate global qubit indexing to the system's local indexing.
        let local_target = qsys.to_local_index(target);
        let local_n_qubits = qsys.qubits.len();
        let local_non_targets: Vec<usize> =
            (0..local_n_qubits).filter(|&i| i != local_target).collect();

        let (measurement, post_measure_density) =
            measure_and_observe_dm(local_target, &qsys.density, local_n_qubits);

        // "Split" density matrix.
        qsys.qubits.remove(local_target);
        qsys.density = reduced_state(&post_measure_density, &local_non_targets, local_n_qubits);

        self.qsystems[target_qsystem] = qsys;

        let collapsed_density = if measurement == 0 {
            dmatrix![cart!(1.0), cart!(0.0);
                     cart!(0.0), cart!(0.0)] // |0><0|
        } else {
            dmatrix![cart!(0.0), cart!(0.0);
                     cart!(0.0), cart!(1.0)] // |1><1|
        };

        self.qsystems.push(QSys {
            density: collapsed_density,
            qubits: vec![target],
        });

        // Write measurement to register.
        let shifted_measurement = measurement << bit_pos;

        if let Value::Int(val) = self.registers[reg] {
            let val_cleared = (val as usize) & !shifted_measurement;
            self.registers[reg] = Value::Int((val_cleared | shifted_measurement) as i32)
        } else {
            self.registers[reg] = Value::Int(shifted_measurement as i32)
        }

        self.pc_mut().increment();
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement_bitstring = collapse_probs(&self.probabilities());

        self.registers[reg] = Value::Int(measurement_bitstring as i32);

        // No entanglement -> one system for each qubit.
        self.qsystems = vec![];

        for qubit in 0..self.circuit.n_qubits() {
            let collapsed_density = if (measurement_bitstring >> qubit) & 1 == 0 {
                dmatrix![cart!(1.0), cart!(0.0);
                         cart!(0.0), cart!(0.0)] // |0><0|
            } else {
                dmatrix![cart!(0.0), cart!(0.0);
                         cart!(0.0), cart!(1.0)] // |1><1|
            };
            self.qsystems.push(QSys {
                density: collapsed_density,
                qubits: vec![qubit],
            })
        }

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
            Instruction::Gate(gate) => self.apply_gate(gate),
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

#[cfg(test)]
mod tests {
    use crate::ext::{equal_to_matrix_c, reduced_state};
    use crate::{
        cart, circuit::Circuit, dm_simulator::DMSimulator, expr_dsl::expr_helpers::r,
        simulator::DebuggableSimulator,
    };
    use nalgebra::{Complex, DMatrix, DVector, dmatrix};

    fn check_probs(sim: &DMSimulator, expected: &DMatrix<Complex<f64>>) {
        let probs = DVector::<Complex<f64>>::from_vec(
            sim.probabilities()
                .iter()
                .map(|&r| cart!(r))
                .collect::<Vec<Complex<f64>>>(),
        );
        assert!(equal_to_matrix_c(&probs, &expected.diagonal(), 0.001));
    }

    #[test]
    fn hch_test() {
        let mut sim = DMSimulator::init(Circuit::new(2).h(0).ch(&[0], 1).into());
        sim.cont();
        let expected_mat = dmatrix![
            cart!(0.5)     , cart!(0.353553), cart!(0.0), cart!(0.353553);
            cart!(0.353553), cart!(0.25)    , cart!(0.0), cart!(0.25);
            cart!(0.0)     , cart!(0.0)     , cart!(0.0), cart!(0.0);
            cart!(0.353553), cart!(0.25)    , cart!(0.0), cart!(0.25);
        ];
        assert!(equal_to_matrix_c(&sim.density(), &expected_mat, 0.001));
        check_probs(&sim, &expected_mat);
    }

    fn print_systems(sim: &DMSimulator) {
        for sys in sim.qsystems.clone() {
            println!("{}", sys);
        }
    }

    #[test]
    fn measure_all_test() {
        for _ in 0..100 {
            let mut sim = DMSimulator::init(
                Circuit::new(5)
                    .new_reg("a")
                    .new_reg("^a")
                    .h(0)
                    .h(1)
                    .h(2)
                    .h(3)
                    .measure("a")
                    .x(0)
                    .x(1)
                    .x(2)
                    .x(3)
                    .measure("^a")
                    .apply_if((r("a") + r("^a")).eq(0b1111))
                    .x(4),
            );
            while let Some(_) = sim.next() {}
            let q4 = reduced_state(&sim.density(), &[4], 5);
            assert!(equal_to_matrix_c(
                &q4,
                &dmatrix![cart!(0.0), cart!(0.0);
                      cart!(0.0), cart!(1.0)],
                0.001
            ));
        }
    }

    #[test]
    fn interleaved_ch_test() {
        let mut sim = DMSimulator::init(
            Circuit::new(4)
                .h(0)
                .ch(&[0], 2)
                .swap(0, 2)
                .h(1)
                .ch(&[1], 3)
                .swap(0, 3)
                .ch(&[2], 1)
                .swap(2, 3)
                .ch(&[0], 3)
                .swap(0, 1)
                .into(),
        );
        sim.cont();
        let mut expected = DMatrix::<Complex<f64>>::zeros(16, 16);

        expected[(0, 0)] = cart!(0.2500000298023224);
        expected[(0, 1)] = cart!(0.1767767071723938);
        expected[(0, 2)] = cart!(0.0625);
        expected[(0, 3)] = cart!(0.0625);
        expected[(0, 6)] = cart!(0.0625);
        expected[(0, 7)] = cart!(-0.0625);
        expected[(0, 8)] = cart!(0.2133883386850357);
        expected[(0, 9)] = cart!(0.0366116538643837);
        expected[(0, 10)] = cart!(-0.0625);
        expected[(0, 11)] = cart!(0.1875000149011612);
        expected[(0, 12)] = cart!(0.2133883386850357);
        expected[(0, 13)] = cart!(0.0366116538643837);
        expected[(0, 14)] = cart!(-0.0625);
        expected[(0, 15)] = cart!(0.0625);

        expected[(1, 0)] = cart!(0.1767767071723938);
        expected[(1, 1)] = cart!(0.125);
        expected[(1, 2)] = cart!(0.04419417679309845);
        expected[(1, 3)] = cart!(0.04419417679309845);
        expected[(1, 6)] = cart!(0.04419417679309845);
        expected[(1, 7)] = cart!(-0.04419417679309845);
        expected[(1, 8)] = cart!(0.1508883386850357);
        expected[(1, 9)] = cart!(0.0258883498609066);
        expected[(1, 10)] = cart!(-0.04419417679309845);
        expected[(1, 11)] = cart!(0.13258253037929535);
        expected[(1, 12)] = cart!(0.1508883386850357);
        expected[(1, 13)] = cart!(0.0258883498609066);
        expected[(1, 14)] = cart!(-0.04419417679309845);
        expected[(1, 15)] = cart!(0.04419417679309845);

        expected[(2, 0)] = cart!(0.0625);
        expected[(2, 1)] = cart!(0.04419417679309845);
        expected[(2, 2)] = cart!(0.015625);
        expected[(2, 3)] = cart!(0.015625);
        expected[(2, 6)] = cart!(0.015625);
        expected[(2, 7)] = cart!(-0.015625);
        expected[(2, 8)] = cart!(0.05334708094596863);
        expected[(2, 9)] = cart!(0.009152913466095924);
        expected[(2, 10)] = cart!(-0.015625);
        expected[(2, 11)] = cart!(0.0468750037252903);
        expected[(2, 12)] = cart!(0.05334708094596863);
        expected[(2, 13)] = cart!(0.009152913466095924);
        expected[(2, 14)] = cart!(-0.015625);
        expected[(2, 15)] = cart!(0.015625);

        expected[(3, 0)] = cart!(0.0625);
        expected[(3, 1)] = cart!(0.04419417679309845);
        expected[(3, 2)] = cart!(0.015625);
        expected[(3, 3)] = cart!(0.015625);
        expected[(3, 6)] = cart!(0.015625);
        expected[(3, 7)] = cart!(-0.015625);
        expected[(3, 8)] = cart!(0.05334708094596863);
        expected[(3, 9)] = cart!(0.009152913466095924);
        expected[(3, 10)] = cart!(-0.015625);
        expected[(3, 11)] = cart!(0.0468750037252903);
        expected[(3, 12)] = cart!(0.05334708094596863);
        expected[(3, 13)] = cart!(0.009152913466095924);
        expected[(3, 14)] = cart!(-0.015625);
        expected[(3, 15)] = cart!(0.015625);

        expected[(6, 0)] = cart!(0.0625);
        expected[(6, 1)] = cart!(0.04419417679309845);
        expected[(6, 2)] = cart!(0.015625);
        expected[(6, 3)] = cart!(0.015625);
        expected[(6, 6)] = cart!(0.015625);
        expected[(6, 7)] = cart!(-0.015625);
        expected[(6, 8)] = cart!(0.05334708094596863);
        expected[(6, 9)] = cart!(0.009152913466095924);
        expected[(6, 10)] = cart!(-0.015625);
        expected[(6, 11)] = cart!(0.0468750037252903);
        expected[(6, 12)] = cart!(0.05334708094596863);
        expected[(6, 13)] = cart!(0.009152913466095924);
        expected[(6, 14)] = cart!(-0.015625);
        expected[(6, 15)] = cart!(0.015625);

        expected[(7, 0)] = cart!(-0.0625);
        expected[(7, 1)] = cart!(-0.04419417679309845);
        expected[(7, 2)] = cart!(-0.015625);
        expected[(7, 3)] = cart!(-0.015625);
        expected[(7, 6)] = cart!(-0.015625);
        expected[(7, 7)] = cart!(0.015625);
        expected[(7, 8)] = cart!(-0.05334708094596863);
        expected[(7, 9)] = cart!(-0.009152913466095924);
        expected[(7, 10)] = cart!(0.015625);
        expected[(7, 11)] = cart!(-0.0468750037252903);
        expected[(7, 12)] = cart!(-0.05334708094596863);
        expected[(7, 13)] = cart!(-0.009152913466095924);
        expected[(7, 14)] = cart!(0.015625);
        expected[(7, 15)] = cart!(-0.015625);

        expected[(8, 0)] = cart!(0.2133883386850357);
        expected[(8, 1)] = cart!(0.1508883386850357);
        expected[(8, 2)] = cart!(0.05334708094596863);
        expected[(8, 3)] = cart!(0.05334708094596863);
        expected[(8, 6)] = cart!(0.05334708094596863);
        expected[(8, 7)] = cart!(-0.05334708094596863);
        expected[(8, 8)] = cart!(0.1821383237838745);
        expected[(8, 9)] = cart!(0.03125);
        expected[(8, 10)] = cart!(-0.05334708094596863);
        expected[(8, 11)] = cart!(0.16004124283790588);
        expected[(8, 12)] = cart!(0.1821383237838745);
        expected[(8, 13)] = cart!(0.03125);
        expected[(8, 14)] = cart!(-0.05334708094596863);
        expected[(8, 15)] = cart!(0.05334708094596863);

        expected[(9, 0)] = cart!(0.0366116538643837);
        expected[(9, 1)] = cart!(0.0258883498609066);
        expected[(9, 2)] = cart!(0.009152913466095924);
        expected[(9, 3)] = cart!(0.009152913466095924);
        expected[(9, 6)] = cart!(0.009152913466095924);
        expected[(9, 7)] = cart!(-0.009152913466095924);
        expected[(9, 8)] = cart!(0.03125);
        expected[(9, 9)] = cart!(0.005361652933061123);
        expected[(9, 10)] = cart!(-0.009152913466095924);
        expected[(9, 11)] = cart!(0.027458740398287773);
        expected[(9, 12)] = cart!(0.03125);
        expected[(9, 13)] = cart!(0.005361652933061123);
        expected[(9, 14)] = cart!(-0.009152913466095924);
        expected[(9, 15)] = cart!(0.009152913466095924);

        expected[(10, 0)] = cart!(-0.0625);
        expected[(10, 1)] = cart!(-0.04419417679309845);
        expected[(10, 2)] = cart!(-0.015625);
        expected[(10, 3)] = cart!(-0.015625);
        expected[(10, 6)] = cart!(-0.015625);
        expected[(10, 7)] = cart!(0.015625);
        expected[(10, 8)] = cart!(-0.05334708094596863);
        expected[(10, 9)] = cart!(-0.009152913466095924);
        expected[(10, 10)] = cart!(0.015625);
        expected[(10, 11)] = cart!(-0.0468750037252903);
        expected[(10, 12)] = cart!(-0.05334708094596863);
        expected[(10, 13)] = cart!(-0.009152913466095924);
        expected[(10, 14)] = cart!(0.015625);
        expected[(10, 15)] = cart!(-0.015625);

        expected[(11, 0)] = cart!(0.1875000149011612);
        expected[(11, 1)] = cart!(0.13258253037929535);
        expected[(11, 2)] = cart!(0.0468750037252903);
        expected[(11, 3)] = cart!(0.0468750037252903);
        expected[(11, 6)] = cart!(0.0468750037252903);
        expected[(11, 7)] = cart!(-0.0468750037252903);
        expected[(11, 8)] = cart!(0.16004124283790588);
        expected[(11, 9)] = cart!(0.027458740398287773);
        expected[(11, 10)] = cart!(-0.0468750037252903);
        expected[(11, 11)] = cart!(0.1406250149011612);
        expected[(11, 12)] = cart!(0.16004124283790588);
        expected[(11, 13)] = cart!(0.027458740398287773);
        expected[(11, 14)] = cart!(-0.0468750037252903);
        expected[(11, 15)] = cart!(0.0468750037252903);

        expected[(12, 0)] = cart!(0.2133883386850357);
        expected[(12, 1)] = cart!(0.1508883386850357);
        expected[(12, 2)] = cart!(0.05334708094596863);
        expected[(12, 3)] = cart!(0.05334708094596863);
        expected[(12, 6)] = cart!(0.05334708094596863);
        expected[(12, 7)] = cart!(-0.05334708094596863);
        expected[(12, 8)] = cart!(0.1821383237838745);
        expected[(12, 9)] = cart!(0.03125);
        expected[(12, 10)] = cart!(-0.05334708094596863);
        expected[(12, 11)] = cart!(0.16004124283790588);
        expected[(12, 12)] = cart!(0.1821383237838745);
        expected[(12, 13)] = cart!(0.03125);
        expected[(12, 14)] = cart!(-0.05334708094596863);
        expected[(12, 15)] = cart!(0.05334708094596863);

        expected[(13, 0)] = cart!(0.0366116538643837);
        expected[(13, 1)] = cart!(0.0258883498609066);
        expected[(13, 2)] = cart!(0.009152913466095924);
        expected[(13, 3)] = cart!(0.009152913466095924);
        expected[(13, 6)] = cart!(0.009152913466095924);
        expected[(13, 7)] = cart!(-0.009152913466095924);
        expected[(13, 8)] = cart!(0.03125);
        expected[(13, 9)] = cart!(0.005361652933061123);
        expected[(13, 10)] = cart!(-0.009152913466095924);
        expected[(13, 11)] = cart!(0.027458740398287773);
        expected[(13, 12)] = cart!(0.03125);
        expected[(13, 13)] = cart!(0.005361652933061123);
        expected[(13, 14)] = cart!(-0.009152913466095924);
        expected[(13, 15)] = cart!(0.009152913466095924);

        expected[(14, 0)] = cart!(-0.0625);
        expected[(14, 1)] = cart!(-0.04419417679309845);
        expected[(14, 2)] = cart!(-0.015625);
        expected[(14, 3)] = cart!(-0.015625);
        expected[(14, 6)] = cart!(-0.015625);
        expected[(14, 7)] = cart!(0.015625);
        expected[(14, 8)] = cart!(-0.05334708094596863);
        expected[(14, 9)] = cart!(-0.009152913466095924);
        expected[(14, 10)] = cart!(0.015625);
        expected[(14, 11)] = cart!(-0.0468750037252903);
        expected[(14, 12)] = cart!(-0.05334708094596863);
        expected[(14, 13)] = cart!(-0.009152913466095924);
        expected[(14, 14)] = cart!(0.015625);
        expected[(14, 15)] = cart!(-0.015625);

        expected[(15, 0)] = cart!(0.0625);
        expected[(15, 1)] = cart!(0.04419417679309845);
        expected[(15, 2)] = cart!(0.015625);
        expected[(15, 3)] = cart!(0.015625);
        expected[(15, 6)] = cart!(0.015625);
        expected[(15, 7)] = cart!(-0.015625);
        expected[(15, 8)] = cart!(0.05334708094596863);
        expected[(15, 9)] = cart!(0.009152913466095924);
        expected[(15, 10)] = cart!(-0.015625);
        expected[(15, 11)] = cart!(0.0468750037252903);
        expected[(15, 12)] = cart!(0.05334708094596863);
        expected[(15, 13)] = cart!(0.009152913466095924);
        expected[(15, 14)] = cart!(-0.015625);
        expected[(15, 15)] = cart!(0.015625);

        assert!(equal_to_matrix_c(&sim.density(), &expected, 0.001));
        check_probs(&sim, &expected);
    }
}
