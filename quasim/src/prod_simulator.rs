use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, PureCircuit, pc::CircuitPc},
    expr_dsl::{BitExpr, BoolExpr},
    ext::collapse,
    gate::Gate,
    instruction::Instruction,
    product_state::ProductState,
    register_file::RegisterFile,
    simulator::{DebuggableSimulator, HybridSimulator, StoredCircuitSimulator},
};
use nalgebra::{Complex, DVector};

#[derive(Debug, Clone)]
pub struct ProdSimulator {
    state: ProductState,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile,
    state_cache: DVector<Complex<f64>>,
}

impl ProdSimulator {
    fn init(circuit: Circuit<HybridCircuit>) -> Self {
        let registers = RegisterFile::from(circuit.registers());

        let mut init_state = DVector::<Complex<f64>>::zeros(1 << circuit.n_qubits());
        init_state[0] = cart!(1.0);

        ProdSimulator {
            state: ProductState::zeros(circuit.n_qubits()),
            circuit: circuit,
            pc: Default::default(),
            registers: registers,
            state_cache: init_state, //TODO: remove cache when state is generic
        }
    }

    fn apply_gate(&mut self, gate: Gate) {
        self.state.apply_gate(gate);
        self.pc_mut().increment();
    }

    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        // Write measurement to register.
        self.registers[reg]
            .write_bit(bit_pos, self.state.collapse_qubit(target))
            .expect("invalid register write");

        self.pc_mut().increment();
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

        self.state_cache = self.state.vector(); //TODO: remove cache when state is generic

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
pub enum ProdSimulatorError {}

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
