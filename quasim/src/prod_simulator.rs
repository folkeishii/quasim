use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit, pc::CircuitPc},
    expr_dsl::{BitExpr, BoolExpr},
    ext::{collapse, expand_matrix_from_gate, measure_and_observe_sv, reduced_state},
    gate::{Gate, GateType},
    instruction::Instruction,
    product_state::{ProductState, SubSystem},
    register_file::{RegisterError, RegisterFile},
    simulator::{DebuggableSimulator, HybridSimulator, StoredCircuitSimulator},
};
use nalgebra::{Complex, DVector, dvector};

#[derive(Debug, Clone)]
pub struct ProdSimulator {
    state: ProductState,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile,
    state_vector_cache: DVector<Complex<f64>>,
}

impl ProdSimulator {
    fn init(circuit: Circuit<HybridCircuit>) -> Self {
        let registers = RegisterFile::from(circuit.registers());
        let init_state = ProductState::zeros(circuit.n_qubits());

        ProdSimulator {
            circuit: circuit,
            pc: Default::default(),
            registers: registers,
            state_vector_cache: init_state.vector(), //TODO: remove cache when state is generic
            state: init_state,
        }
    }

    pub fn get_state(&self) -> &ProductState {
        &self.state
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
            let sys_idx = self.state.system_of_qubit(qubit);
            if !gate_systems.contains(&sys_idx) {
                gate_systems.push(sys_idx);
            }
        }

        let mut sys = self.state.systems[gate_systems[0]].clone();

        let gate_acts_on_several_systems = gate_systems.len() > 1;
        let is_swap_gate = gate.get_type() == GateType::SWAP && controls.is_empty();

        if gate_acts_on_several_systems && is_swap_gate {
            // Regular swaps do not result in entanglement.
            // Only change global qubit index.
            let sys1_local_target = sys.local_index(targets[0]);
            let mut sys2 = self.state.systems[gate_systems[1]].clone();
            let sys2_local_target = sys2.local_index(targets[1]);

            // Swap.
            let temp = sys.qubits[sys1_local_target];
            sys.qubits[sys1_local_target] = sys2.qubits[sys2_local_target];
            sys2.qubits[sys2_local_target] = temp;

            self.state.systems[gate_systems[0]] = sys;
            self.state.systems[gate_systems[1]] = sys2;
            return;
        } else if gate_acts_on_several_systems {
            // Combine all systems acted on.
            sys = gate_systems
                .iter()
                .skip(1)
                .map(|&qs_idx| self.state.systems[qs_idx].clone())
                .fold(sys, |acc, qs| acc.combine(&qs));
        }

        // Translate global qubit indexing to the system's local indexing.
        let local_targets: Vec<usize> = targets.iter().map(|&t| sys.local_index(t)).collect();

        if is_swap_gate {
            // Swap qubit indecies, no need for matrix multiplication.
            sys.qubits.swap(local_targets[0], local_targets[1]);
            self.state.systems[gate_systems[0]] = sys;
            return;
        }

        // Continue to translate global qubit indexing to the system's local indexing.
        let local_controls: Vec<usize> = controls.iter().map(|&t| sys.local_index(t)).collect();
        let local_n_qubits = sys.qubits.len();
        let local_gate = Gate::new(gate.get_type(), &local_controls, &local_targets).unwrap();

        // Apply the gate to the system's state vector using: |s>` == U|s>
        let mat = expand_matrix_from_gate(&local_gate, local_n_qubits);
        sys.state = mat * sys.state;

        if gate_acts_on_several_systems {
            // Remove systems that were combined.
            self.state.systems = self
                .state
                .systems
                .iter()
                .enumerate()
                .filter(|(i, _)| !gate_systems.contains(i))
                .map(|(_, s)| s.clone())
                .collect();

            // Add combined system.
            self.state.systems.push(sys);
            return;
        }
        // Update system acted on.
        self.state.systems[gate_systems[0]] = sys;
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

        self.pc_mut().increment();

        let target_system = self.state.system_of_qubit(target);
        let mut sys = self.state.systems[target_system].clone();

        // Translate global qubit indexing to the system's local indexing.
        let local_target = sys.local_index(target);
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
        self.state.systems[target_system] = sys;

        let collapsed_state = if measurement == 0 {
            dvector![cart!(1.0), cart!(0.0)] // |0>
        } else {
            dvector![cart!(0.0), cart!(1.0)] // |1>
        };

        self.state.systems.push(SubSystem {
            state: collapsed_state,
            qubits: vec![target],
        });

        // Write measurement to register.
        self.registers[reg]
            .write_bit(bit_pos, measurement)
            .expect("invalid register write");
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement_bitstring = collapse(&self.state.vector().as_slice());

        self.registers[reg].write(measurement_bitstring);

        self.state = ProductState::from_bitstring(measurement_bitstring, self.n_qubits());

        self.pc_mut().increment();
    }

    fn jump(&mut self, label_pc: usize) {
        self.pc_mut().jump(label_pc);
    }

    fn jump_if(&mut self, expr: &BoolExpr, label_pc: usize) {
        if expr.eval(&self.registers) {
            self.jump(label_pc)
        } else {
            self.pc_mut().increment()
        }
    }

    fn assign(&mut self, expr: &BitExpr, reg: &str) {
        let value = expr.eval(&self.registers);
        self.registers[reg].write(value);
        self.pc_mut().increment();
    }

    fn pc(&self) -> &CircuitPc {
        &self.pc
    }

    fn pc_mut(&mut self) -> &mut CircuitPc {
        &mut self.pc
    }
}

impl TryFrom<Circuit<PureCircuit>> for ProdSimulator {
    type Error = ProdSimulatorError;

    fn try_from(value: Circuit<PureCircuit>) -> Result<Self, Self::Error> {
        Self::try_from(Circuit::<HybridCircuit>::from(value.into()))
    }
}

impl TryFrom<Circuit<HybridCircuit>> for ProdSimulator {
    type Error = ProdSimulatorError;

    fn try_from(value: Circuit<HybridCircuit>) -> Result<Self, Self::Error> {
        let circuit = value;

        let sim = Self::init(circuit);

        Ok(sim)
    }
}

impl HybridSimulator for ProdSimulator {
    fn registers(&self) -> &RegisterFile {
        &self.registers
    }
}

impl DebuggableSimulator for ProdSimulator {
    type Storage = DVector<Complex<f64>>;
    type State = Complex<f64>;

    fn collapse_peek(&self) -> usize {
        collapse(&self.state.vector().as_slice())
    }

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

        self.state_vector_cache = self.state.vector(); //TODO: remove cache when state is generic

        true
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<Instruction>) {
        (self.pc(), self.circuit.instruction(self.pc()))
    }

    fn current_state(&self) -> &DVector<Complex<f64>> {
        &self.state_vector_cache //TODO: remove cache when state is generic
    }

    fn double_ended(&self) -> bool {
        false
    }
}
impl StoredCircuitSimulator for ProdSimulator {
    type B = HybridCircuit;
    fn circuit(&self) -> &Circuit<HybridCircuit> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<HybridCircuit> {
        &mut self.circuit
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ProdSimulatorError {
    #[error("{0}")]
    RegisterError(#[from] RegisterError),
}

#[cfg(test)]
mod tests {
    use crate::common_test;
    use crate::prod_simulator::ProdSimulator;

    #[test]
    fn hybrid_test() {
        common_test::hybrid_test::<ProdSimulator>();
    }

    #[test]
    fn register_test() {
        common_test::register_test::<ProdSimulator>();
    }

    #[test]
    fn double_sub() {
        common_test::double_sub::<ProdSimulator>();
    }

    #[test]
    fn deep_sub() {
        common_test::deep_sub::<ProdSimulator>();
    }

    #[test]
    fn deep_ctrl_sub() {
        common_test::deep_ctrl_sub::<ProdSimulator>();
    }

    #[test]
    fn interleaved() {
        common_test::interleaved::<ProdSimulator>();
    }

    #[test]
    fn mid_measure_all() {
        common_test::mid_measure_all::<ProdSimulator>();
    }

    #[test]
    fn mid_measure_bit() {
        common_test::mid_measure_bit::<ProdSimulator>();
    }
}
