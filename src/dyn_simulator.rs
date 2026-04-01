use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit, pc::CircuitPc},
    expr_dsl::{Expr, Value},
    ext::{collapse, expand_matrix_from_gate, measure_and_observe_sv, reduced_state, swap_matrix},
    gate::{Gate, GateType},
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{DebuggableSimulator, HybridSimulator, StoredCircuitSimulator},
};
use nalgebra::{Complex, DMatrix, DVector, dvector};

/// A system of potentially entangled qubits.
#[derive(Debug, Clone)]
struct EntSys {
    state: DVector<Complex<f64>>,
    qubits: Vec<usize>,
}

impl std::fmt::Display for EntSys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Qubits: {:?}\n Density: {}", self.qubits, self.state)
    }
}

impl EntSys {
    /// Concatinates qubit-lists and "tensors" state vectors.
    fn add_system(&self, rhs: &Self) -> Self {
        let mut qubits = self.qubits.clone();
        qubits.extend(rhs.qubits.clone());
        Self {
            state: rhs.state.kronecker(&self.state),
            qubits: qubits,
        }
    }

    fn to_local_index(&self, global_index: usize) -> usize {
        let Some(local_index) = self.qubits.iter().position(|&q| q == global_index) else {
            panic!("Qubit is not member of system!")
        };
        local_index
    }

    /// Insersion sort by qubit index and swaps on state vector :w
    /// accordingly.
    fn sort(&mut self) {
        let n_qubits = self.qubits.len();
        let mut i = 1;
        while i < n_qubits {
            let mut j = i;
            while j > 0 && self.qubits[j - 1] > self.qubits[j] {
                // Sort the list of qubit indecies.
                self.qubits.swap(j, j - 1);

                // Sort the state vector.
                self.state = swap_matrix(&[], j, j - 1, n_qubits) * self.state.clone();

                j -= 1;
            }
            i += 1;
        }
    }
}

fn system_product(systems: &[EntSys]) -> EntSys {
    systems.into_iter().fold(
        EntSys {
            state: dvector![cart!(1.0)],
            qubits: vec![],
        },
        |acc, sys| acc.add_system(&sys),
    )
}

#[derive(Debug, Clone)]
pub struct DynSimulator {
    systems: Vec<EntSys>,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile<Value>,
    state_cache: DVector<Complex<f64>>,
}

impl DynSimulator {
    fn init(circuit: Circuit<HybridCircuit>) -> Self {
        // Initial state assumed to be |000..>
        // No entanglement -> one system for each qubit.

        let mut init_sys = vec![];

        for i in 0..circuit.n_qubits() {
            init_sys.push(EntSys {
                state: dvector![cart!(1.0), cart!(0.0)], // |0>
                qubits: vec![i],
            });
        }

        let registers = RegisterFile::from(circuit.registers());

        let mut init_state = DVector::<Complex<f64>>::zeros(1 << circuit.n_qubits());
        init_state[0] = cart!(1.0);

        DynSimulator {
            systems: init_sys,
            circuit: circuit,
            pc: Default::default(),
            registers: registers,
            state_cache: init_state, //TODO: remove cache when state is generic
        }
    }

    fn find_system_of_qubit(&self, qubit: usize) -> usize {
        let Some(sys_idx) = self.systems.iter().position(|s| s.qubits.contains(&qubit)) else {
            panic!("Qubit is not member of any system!")
        };
        sys_idx
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
         *      (Exception: SWAP gates, they do not
         *      cause entanglement)
         *
         * */

        self.pc_mut().increment();

        let controls = gate.get_controls();
        let targets = gate.get_targets();

        // Qubits that gate acts on.
        let mut gate_qubits = controls.clone();
        gate_qubits.extend(&targets);

        // Systems that gate acts on.
        let mut gate_systems = vec![];
        for qubit in gate_qubits {
            let sys_idx = self.find_system_of_qubit(qubit);
            if !gate_systems.contains(&sys_idx) {
                gate_systems.push(sys_idx);
            }
        }

        let mut sys = self.systems[gate_systems[0]].clone();

        let gate_acts_on_several_systems = gate_systems.len() > 1;
        let is_swap_gate = gate.get_type() == GateType::SWAP && controls.is_empty();

        if gate_acts_on_several_systems && is_swap_gate {
            // Regular swaps do not result in entanglement.
            // Only change global qubit index.
            let sys1_local_target = sys.to_local_index(targets[0]);
            let mut sys2 = self.systems[gate_systems[1]].clone();
            let sys2_local_target = sys2.to_local_index(targets[1]);

            // Swap.
            let temp = sys.qubits[sys1_local_target];
            sys.qubits[sys1_local_target] = sys2.qubits[sys2_local_target];
            sys2.qubits[sys2_local_target] = temp;

            self.systems[gate_systems[0]] = sys;
            self.systems[gate_systems[1]] = sys2;
            return;
        } else if gate_acts_on_several_systems {
            // Combine all systems acted on.
            sys = gate_systems
                .iter()
                .skip(1)
                .map(|&qs_idx| self.systems[qs_idx].clone())
                .fold(sys, |acc, qs| acc.add_system(&qs));
        }

        // Translate global qubit indexing to the system's local indexing.
        let local_targets: Vec<usize> = targets.iter().map(|&t| sys.to_local_index(t)).collect();

        if is_swap_gate {
            // Swap qubit indecies, no need for matrix multiplication.
            sys.qubits.swap(local_targets[0], local_targets[1]);
            self.systems[gate_systems[0]] = sys;
            return;
        }

        // Continue to translate global qubit indexing to the system's local indexing.
        let local_controls: Vec<usize> = controls.iter().map(|&t| sys.to_local_index(t)).collect();
        let local_n_qubits = sys.qubits.len();
        let local_gate = Gate::new(gate.get_type(), &local_controls, &local_targets).unwrap();

        // Apply the gate to the system's state vector using: |s>` == U|s>
        let mat = expand_matrix_from_gate(&local_gate, local_n_qubits);
        sys.state = mat * sys.state;

        if gate_acts_on_several_systems {
            // Remove systems that were combined.
            self.systems = self
                .systems
                .iter()
                .enumerate()
                .filter(|(i, _)| !gate_systems.contains(i))
                .map(|(_, s)| s.clone())
                .collect();

            // Add combined system.
            self.systems.push(sys);
            return;
        }
        // Update system acted on.
        self.systems[gate_systems[0]] = sys;
    }
    /// The density matrix of the whole system.
    fn density_matrix(&self) -> DMatrix<Complex<f64>> {
        let v = self.state_vector();
        let v_adj = v.adjoint();
        v * v_adj // |s><s|
    }

    /// The state vector of the whole system.
    /// Only works for pure states
    fn state_vector(&self) -> DVector<Complex<f64>> {
        let mut tot_sys = system_product(&self.systems);

        tot_sys.sort();
        tot_sys.state
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

        let target_system = self.find_system_of_qubit(target);
        let mut sys = self.systems[target_system].clone();

        // Translate global qubit indexing to the system's local indexing.
        let local_target = sys.to_local_index(target);
        let local_n_qubits = sys.qubits.len();
        let local_non_targets: Vec<usize> =
            (0..local_n_qubits).filter(|&i| i != local_target).collect();

        let (measurement, post_measure_state) =
            measure_and_observe_sv(local_target, &sys.state, local_n_qubits);

        // "Split" state.
        sys.qubits.remove(local_target);
        let post_measure_state_adj = post_measure_state.adjoint();
        let density = post_measure_state * post_measure_state_adj; //|s><s|
        sys.state = DVector::from(
            reduced_state(&density, &local_non_targets, local_n_qubits)
                .symmetric_eigen()
                .eigenvectors
                .column(0),
        );
        self.systems[target_system] = sys;

        let collapsed_state = if measurement == 0 {
            dvector![cart!(1.0), cart!(0.0)] // |0>
        } else {
            dvector![cart!(0.0), cart!(1.0)] // |1>
        };

        self.systems.push(EntSys {
            state: collapsed_state,
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
        let measurement_bitstring = collapse(&self.state_vector().as_slice());

        self.registers[reg] = Value::Int(measurement_bitstring as i32);

        // No entanglement -> one system for each qubit.
        self.systems = vec![];

        for qubit in 0..self.circuit.n_qubits() {
            let collapsed_state = if (measurement_bitstring >> qubit) & 1 == 0 {
                dvector![cart!(1.0), cart!(0.0)] // |0>
            } else {
                dvector![cart!(0.0), cart!(1.0)] // |1>
            };
            self.systems.push(EntSys {
                state: collapsed_state,
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

impl TryFrom<Circuit<PureCircuit>> for DynSimulator {
    type Error = DynSimulatorError;

    fn try_from(value: Circuit<PureCircuit>) -> Result<Self, Self::Error> {
        Self::try_from(Circuit::<HybridCircuit>::from(value.into()))
    }
}

impl TryFrom<Circuit<HybridCircuit>> for DynSimulator {
    type Error = DynSimulatorError;

    fn try_from(value: Circuit<HybridCircuit>) -> Result<Self, Self::Error> {
        let circuit = value;

        let sim = Self::init(circuit);

        Ok(sim)
    }
}

impl HybridSimulator<Value> for DynSimulator {
    fn registers(&self) -> &RegisterFile<Value> {
        &self.registers
    }
}

impl DebuggableSimulator for DynSimulator {
    fn next(&mut self) -> bool {
        let Some(inst) = self.circuit.instruction(self.pc()) else {
            return false;
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

        self.state_cache = self.state_vector(); //TODO: remove cache when state is generic

        true
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>) {
        (self.pc(), self.circuit.instruction(self.pc()))
    }

    fn current_state(&self) -> &DVector<Complex<f64>> {
        &self.state_cache //TODO: remove cache when state is generic
    }

    fn double_ended(&self) -> bool {
        false
    }
}
impl StoredCircuitSimulator for DynSimulator {
    type B = HybridCircuit;
    fn circuit(&self) -> &Circuit<HybridCircuit> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<HybridCircuit> {
        &mut self.circuit
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum DynSimulatorError {
    #[error("Measurement mid-circuit")]
    MidCircuitMeasurement,
}

#[cfg(test)]
mod tests {
    use crate::common_test;
    use crate::ext::{equal_to_matrix_c, reduced_state};
    use crate::{
        cart, circuit::Circuit, dyn_simulator::DynSimulator, expr_dsl::expr_helpers::r,
        simulator::DebuggableSimulator,
    };
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn hybrid_test() {
        common_test::hybrid_test::<DynSimulator>();
    }

    #[test]
    fn register_test() {
        common_test::register_test::<DynSimulator>();
    }

    #[test]
    fn double_sub() {
        common_test::double_sub::<DynSimulator>();
    }

    #[test]
    fn deep_sub() {
        common_test::deep_sub::<DynSimulator>();
    }

    #[test]
    fn deep_ctrl_sub() {
        common_test::deep_ctrl_sub::<DynSimulator>();
    }

    fn print_systems(sim: &DynSimulator) {
        for sys in sim.systems.clone() {
            println!("{}", sys);
        }
    }

    #[test]
    fn measure_all_test() {
        let mut sim = DynSimulator::init(
            Circuit::new(5)
                .new_reg("a")
                .new_reg("~a")
                .h(0)
                .h(1)
                .h(2)
                .h(3)
                .measure("a")
                .x(0)
                .x(1)
                .x(2)
                .x(3)
                .measure("~a")
                .apply_if((r("a") + r("~a")).eq(0b1111))
                .x(4),
        );
        while sim.next() {}
        let q4 = reduced_state(&sim.density_matrix(), &[4], 5);
        assert!(equal_to_matrix_c(
            &q4,
            &dmatrix![cart!(0.0), cart!(0.0);
                      cart!(0.0), cart!(1.0)],
            0.001
        ));
    }

    #[test]
    fn measure_bit_test() {
        let mut sim = DynSimulator::init(
            Circuit::new(5)
                .new_reg("a")
                .new_reg("~a")
                .h(0)
                .h(1)
                .h(2)
                .h(3)
                .measure_bit(0, ("a", 0))
                .measure_bit(1, ("a", 1))
                .measure_bit(2, ("a", 2))
                .measure_bit(3, ("a", 3))
                .x(0)
                .x(1)
                .x(2)
                .x(3)
                .measure_bit(0, ("~a", 0))
                .measure_bit(1, ("~a", 1))
                .measure_bit(2, ("~a", 2))
                .measure_bit(3, ("~a", 3))
                .apply_if((r("a") + r("~a")).eq(0b1111))
                .x(4),
        );
        while sim.next() {}
        let q4 = reduced_state(&sim.density_matrix(), &[4], 5);
        assert!(equal_to_matrix_c(
            &q4,
            &dmatrix![cart!(0.0), cart!(0.0);
                      cart!(0.0), cart!(1.0)],
            0.001
        ));
    }

    #[test]
    fn interleaved_ch_test() {
        let mut sim = DynSimulator::init(
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
        while sim.next() {}

        let expected = dvector![
            cart!(0.5000000293365844),
            cart!(0.35355340368276855),
            cart!(0.12500000042912138),
            cart!(0.12500000042912138),
            cart!(0.0),
            cart!(0.0),
            cart!(0.12500000042912138),
            cart!(-0.12500000042912138),
            cart!(0.4267766656533139),
            cart!(0.07322330885223931),
            cart!(-0.12500000042912138),
            cart!(0.37500001339455813),
            cart!(0.4267766656533139),
            cart!(0.07322330885223931),
            cart!(-0.12500000042912138),
            cart!(0.12500000042912138),
        ];

        assert!(equal_to_matrix_c(&sim.state_vector(), &expected, 0.001));
    }
}
