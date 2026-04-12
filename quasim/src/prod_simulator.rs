use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit, pc::CircuitPc},
    expr_dsl::{BitExpr, BoolExpr},
    ext::{collapse, get_u_matrix2, measure_state_vector, schmitt_trace},
    gate::{Gate, GateType, QBits},
    instruction::Instruction,
    product_state::{ProductState, SubSystem},
    register_file::{RegisterError, RegisterFile},
    simulator::{DebuggableSimulator, HybridSimulator, StoredCircuitSimulator},
};
use nalgebra::{Complex, DVector, Matrix2};

#[derive(Debug, Clone)]
pub struct ProdSimulator {
    product_state: ProductState,
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
            product_state: init_state,
        }
    }

    pub fn get_state(&self) -> &ProductState {
        &self.product_state
    }

    /// Checks that all control bits are 1
    fn controls_active(i: usize, controls: QBits) -> bool {
        let control_mask = controls.get_bitstring();
        (i & control_mask) == control_mask
    }

    /// Checks that all target bits are 0
    fn is_block_base(i: usize, targets: QBits) -> bool {
        let target_mask = targets.get_bitstring();
        (i & target_mask) == 0
    }

    // 0 1
    // 1 0
    #[inline(always)]
    fn apply_x(state_vector: &mut DVector<Complex<f64>>, base_index: usize, target: QBits) {
        state_vector
            .as_mut_slice()
            .swap(base_index, base_index | target.get_bitstring());
    }

    // 0 -i
    // i  0
    #[inline(always)]
    fn apply_y(state_vector: &mut DVector<Complex<f64>>, base_index: usize, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let state = state_vector.as_mut_slice();
        let a = state[base_index];
        let b = state[flipped_index];

        state[base_index] = cart!(b.im, -b.re);
        state[flipped_index] = cart!(-a.im, a.re);
    }

    // 1  0
    // 0 -1
    #[inline(always)]
    fn apply_z(state_vector: &mut DVector<Complex<f64>>, base_index: usize, target: QBits) {
        let i = base_index | target.get_bitstring();
        let amp = &mut state_vector[i];

        amp.re = -amp.re;
        amp.im = -amp.im;
    }

    #[inline(always)]
    fn apply_h(state_vector: &mut DVector<Complex<f64>>, base_index: usize, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let state = state_vector.as_mut_slice();
        let a = state[base_index];
        let b = state[flipped_index];
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;

        state[base_index] = (a + b) * inv_sqrt2;
        state[flipped_index] = (a - b) * inv_sqrt2;
    }

    #[inline(always)]
    fn apply_s(state_vector: &mut DVector<Complex<f64>>, base_index: usize, target: QBits) {
        let i = base_index | target.get_bitstring();
        let amp = state_vector[i];

        state_vector[i].re = -amp.im;
        state_vector[i].im = amp.re;
    }

    #[inline(always)]
    fn apply_swap(state_vector: &mut DVector<Complex<f64>>, base_index: usize, targets: QBits) {
        let t0 = targets.get_indices()[0];
        let t1 = targets.get_indices()[1];

        let i01 = base_index | (1 << t0);
        let i10 = base_index | (1 << t1);

        state_vector.as_mut_slice().swap(i01, i10);
    }

    #[inline(always)]
    fn apply_unitary2(
        state_vector: &mut DVector<Complex<f64>>,
        base_index: usize,
        u: &Matrix2<Complex<f64>>,
        target: QBits,
    ) {
        let flipped_index = base_index | target.get_bitstring();
        let a = state_vector[base_index];
        let b = state_vector[flipped_index];

        state_vector[base_index] = u[(0, 0)] * a + u[(0, 1)] * b;
        state_vector[flipped_index] = u[(1, 0)] * a + u[(1, 1)] * b;
    }

    fn gate(state_vector: &mut DVector<Complex<f64>>, gate: &Gate) {
        let controls = gate.get_control_bits();
        let targets = gate.get_target_bits();
        let n = state_vector.len();

        // No parallelization
        // State vector is length 2^n , n=num qubits
        for i in 0..n {
            if !Self::is_block_base(i, targets) {
                continue;
            }

            if !Self::controls_active(i, controls) {
                continue;
            }

            match gate.get_type() {
                GateType::X => Self::apply_x(state_vector, i, targets),
                GateType::Y => Self::apply_y(state_vector, i, targets),
                GateType::Z => Self::apply_z(state_vector, i, targets),
                GateType::H => Self::apply_h(state_vector, i, targets),
                GateType::S => Self::apply_s(state_vector, i, targets),
                GateType::SWAP => Self::apply_swap(state_vector, i, targets),
                GateType::U(theta, phi, lambda) => Self::apply_unitary2(
                    state_vector,
                    i,
                    &get_u_matrix2(theta, phi, lambda),
                    targets,
                ),
            }
        }
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
            let sys_idx = self.product_state.system_of_qubit(qubit);
            if !gate_systems.contains(&sys_idx) {
                gate_systems.push(sys_idx);
            }
        }

        let mut sys = self.product_state[gate_systems[0]].clone();

        let gate_acts_on_several_systems = gate_systems.len() > 1;
        let is_non_controlled_swap_gate = gate.get_type() == GateType::SWAP && controls.is_empty();

        if gate_acts_on_several_systems && is_non_controlled_swap_gate {
            // Regular swaps do not result in entanglement.
            // Only change global qubit index.
            let sys1_local_target = sys.local_index(targets[0]);
            let mut sys2 = self.product_state[gate_systems[1]].clone();
            let sys2_local_target = sys2.local_index(targets[1]);

            // Swap.
            sys.qubits_mut()[sys1_local_target] = targets[1];
            sys2.qubits_mut()[sys2_local_target] = targets[0];

            self.product_state[gate_systems[0]] = sys;
            self.product_state[gate_systems[1]] = sys2;
            return;
        } else if gate_acts_on_several_systems {
            // Combine all systems acted on.
            sys = gate_systems
                .clone()
                .into_iter()
                .skip(1)
                .map(|qs_idx| self.product_state[qs_idx].clone())
                .fold(sys, |acc, qs| acc * qs);
        }

        // Translate global qubit indexing to the system's local indexing.
        let local_targets: Vec<usize> = targets.iter().map(|&t| sys.local_index(t)).collect();

        if is_non_controlled_swap_gate {
            // Swap qubit indecies, no need for matrix multiplication.
            sys.qubits_mut()[local_targets[0]] = targets[1];
            sys.qubits_mut()[local_targets[1]] = targets[0];
            self.product_state[gate_systems[0]] = sys;
            return;
        }

        // Continue to translate global qubit indexing to the system's local indexing.
        let local_controls: Vec<usize> = controls.iter().map(|&c| sys.local_index(c)).collect();
        //let local_n_qubits = sys.n_qubits();
        let local_gate = Gate::new(gate.get_type(), &local_controls, &local_targets).unwrap();

        // Apply the gate to the system's state vector using: |s>` == U|s>
        Self::gate(sys.state_vector_mut(), &local_gate);
        //let mat = expand_matrix_from_gate(&local_gate, local_n_qubits);
        //*sys.state_vector_mut() = mat * sys.state_vector();

        if gate_acts_on_several_systems {
            // Remove systems that were combined.
            self.product_state = ProductState::from(
                self.product_state
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !gate_systems.contains(i))
                    .map(|(_, s)| s.clone())
                    .collect::<Vec<SubSystem>>(),
            );

            // Add combined system.
            self.product_state.push(sys);
            return;
        }
        // Update system acted on.
        self.product_state[gate_systems[0]] = sys;
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
         *                      p` == |0><0| * Tr_A(p)
         * */

        self.pc_mut().increment();

        let target_system = self.product_state.system_of_qubit(target);
        let mut sys = self.product_state[target_system].clone();

        // Translate global qubit indexing to the system's local indexing.
        let local_target = sys.local_index(target);
        let local_n_qubits = sys.n_qubits();

        let measurement =
            measure_state_vector(sys.state_vector_mut(), local_target, local_n_qubits);

        // "Split" state.
        sys.qubits_mut().remove(local_target);
        *sys.state_vector_mut() =
            schmitt_trace(sys.state_vector(), &[local_target], local_n_qubits);

        self.product_state[target_system] = sys;

        if measurement == 0 {
            self.product_state.push(SubSystem::zero(target));
        } else {
            self.product_state.push(SubSystem::one(target));
        };

        // Write measurement to register.
        self.registers[reg]
            .write_bit(bit_pos, measurement)
            .expect("invalid register write");
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement_bitstring = collapse(&self.product_state.vector().as_slice());

        self.registers[reg].write(measurement_bitstring);

        self.product_state = ProductState::from_bitstring(measurement_bitstring, self.n_qubits());

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
        collapse(&self.product_state.vector().as_slice())
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

        self.state_vector_cache = self.product_state.vector(); //TODO: remove cache when state is generic

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
