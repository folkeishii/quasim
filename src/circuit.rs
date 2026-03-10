use std::{
    collections::{HashMap, HashSet},
    f64::consts::PI,
};
pub mod breakpoint;
pub mod pc;

use crate::{
    circuit::{
        breakpoint::{Breakpoint, BreakpointList, IEBreakpoint},
        pc::CircuitPc,
    },
    expr_dsl::Expr,
    gate::{Gate, GateType},
    instruction::Instruction,
};
mod qasm_parse;

use log::{trace, warn};
use oq3_syntax::{SourceFile, ast::AstNode};
use std::fs::read_to_string;

pub use qasm_parse::*;

#[derive(Debug, Clone)]
pub struct Circuit {
    instructions: Vec<Instruction>,
    n_qubits: usize,
    labels: HashMap<String, usize>,
    unresolved_labels: Vec<(String, usize)>,
    breakpoints: BreakpointList,
    registers: HashSet<String>,
}

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            instructions: Vec::<Instruction>::default(),
            n_qubits: n_qubits,
            registers: HashSet::new(),
            labels: HashMap::new(),
            unresolved_labels: Vec::new(),
            breakpoints: Default::default(),
        }
    }

    pub fn new_reg<I: Into<String>>(mut self, name: I) -> Self {
        self.registers.insert(name.into());
        self
    }

    pub fn valid_pc(&self, circuit_pc: &CircuitPc) -> bool {
        circuit_pc.pc() <= self.instructions().len()
    }

    pub fn instruction(&self, circuit_pc: &CircuitPc) -> Option<Instruction> {
        match self.instructions().get(circuit_pc.pc()) {
            Some(Instruction::Gate(gate)) => {
                Some(Instruction::Gate(gate.clone() << circuit_pc.lsq()))
            }
            Some(Instruction::MeasureBit(target, register)) => Some(Instruction::MeasureBit(
                *target << circuit_pc.lsq(),
                register.clone(),
            )),
            rst => rst.cloned(),
        }
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn registers(&self) -> &HashSet<String> {
        &self.registers
    }

    pub fn has_unresolved_labels(&self) -> bool {
        !self.unresolved_labels.is_empty()
    }

    pub fn from_qasm_file(file_name: &str) -> Result<Self, QASMParseError> {
        let file_string = read_to_string(file_name)?;
        let parsed_source = SourceFile::parse(&file_string);
        let parse_tree: SourceFile = parsed_source.tree();
        trace!(
            "Found {} QASM statements",
            parse_tree.statements().collect::<Vec<_>>().len()
        );
        let syntax_errors = parsed_source.errors();
        if syntax_errors.len() > 0 {
            warn!(
                "Found {} QASM parse errors:\n{:?}\n",
                syntax_errors.len(),
                syntax_errors
            );
        }

        // First pass: count the number of qubits
        let n_qubits = count_qubits_from_syntax_tree(parse_tree.syntax())?;

        // Second pass: build the circuit by applying gates
        let circuit = apply_gates_from_syntax_tree(Circuit::new(n_qubits), parse_tree.syntax())?;

        return Ok(circuit);
    }

    // Builder methods

    pub fn x(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::X, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cx(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::X, controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn y(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Y, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cy(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Y, controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn z(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Z, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cz(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Z, controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn h(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::H, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn ch(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::H, controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn swap(mut self, target1: usize, target2: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::SWAP, &[], &[target1, target2]).unwrap(),
        ));
        self
    }

    pub fn cswap(mut self, controls: &[usize], target1: usize, target2: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::SWAP, controls, &[target1, target2]).unwrap(),
        ));
        self
    }

    pub fn u(mut self, theta: f64, phi: f64, lambda: f64, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, phi, lambda), &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cu(
        mut self,
        theta: f64,
        phi: f64,
        lambda: f64,
        controls: &[usize],
        target: usize,
    ) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, phi, lambda), controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn s(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::S, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cs(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::S, controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn rx(mut self, theta: f64, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, -PI / 2.0, PI / 2.0), &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn crx(mut self, theta: f64, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, -PI / 2.0, PI / 2.0), controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn ry(mut self, theta: f64, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, 0.0, 0.0), &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cry(mut self, theta: f64, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, 0.0, 0.0), controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn rz(mut self, theta: f64, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(0.0, 0.0, theta), &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn crz(mut self, theta: f64, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(0.0, 0.0, theta), controls, &[target]).unwrap(),
        ));
        self
    }

    // Classical instructions

    pub fn measure_bit(mut self, target: usize, reg: (&str, usize)) -> Self {
        self.instructions
            .push(Instruction::MeasureBit(target, (reg.0.into(), reg.1)));
        self
    }

    /// Measure multiple bits into a register
    ///
    /// Example:
    /// ```ignore
    /// measure_bits(&[2,1,3], "reg")
    /// // Is equivalent to
    /// measure_bit(2, ("reg", 0))
    /// measure_bit(1, ("reg", 1))
    /// measure_bit(3, ("reg", 2))
    /// ```
    pub fn measure_bits(mut self, targets: &[usize], reg: &str) -> Self {
        for (i, target) in targets.iter().enumerate() {
            self = self.measure_bit(*target, (reg, i))
        }
        self
    }

    /// Measures all qubits into a register
    pub fn measure(mut self, reg: &str) -> Self {
        self.instructions.push(Instruction::MeasureAll(reg.into()));
        self
    }

    pub fn jump(mut self, label: String) -> Self {
        let circuit_pc = match self.try_to_resolve_label(label) {
            Some(circuit_pc) => circuit_pc.clone(),
            None => 0, // Placeholder pc
        };

        self.instructions.push(Instruction::Jump(circuit_pc));
        self
    }

    pub fn jump_if(mut self, expr: Expr, label: String) -> Self {
        let circuit_pc = match self.try_to_resolve_label(label) {
            Some(circuit_pc) => circuit_pc.clone(),
            None => 0, // Placeholder pc
        };

        self.instructions
            .push(Instruction::JumpIf(expr, circuit_pc));
        self
    }

    /// Conditionally apply whichever instruction that comes after
    pub fn apply_if(mut self, expr: Expr) -> Self {
        self.instructions
            .push(Instruction::JumpIf(!expr, self.instructions.len() + 2));
        self
    }

    pub fn reset(self, target: usize) -> Self {
        self.new_reg("_reset")
            .measure_bit(target, ("_reset", 0))
            .apply_if(Expr::Reg("_reset".to_owned()).eq(1))
            .x(target)
    }

    // takes register nr directly for now
    pub fn assign(mut self, reg: String, expr: Expr) -> Self {
        if !self.registers.contains(&reg) {
            panic!(
                "Tried to assign to nonexistent register with name '{}'.",
                reg
            )
        }
        self.instructions.push(Instruction::Assign(expr, reg));
        self
    }

    /// Appends a circuit implementing the quantum Fourier transform.
    /// Targets are normally specified in order of least significance,
    /// for example [0,1,2,3,4].
    pub fn qft(mut self, targets: &[usize]) -> Self {
        /* This implementation is taken from Mike & Ike chapter 5.1.
         * Note that, due to our chosen convention, the circuit will
         * be the same as figure 5.1 but "upside down".
         * */

        let n = targets.len();

        for i in (0..n).rev() {
            self = self.h(targets[i]);

            let mut control: isize = i as isize - 1;
            for k in 2..(i + 2) {
                let theta = PI / (1 << (k - 1)) as f64;
                self = self.crz(theta, &[targets[control as usize]], targets[i]);
                control -= 1;
            }
        }

        // Reverse order of qubits. (not shown in figure 5.1)
        for i in 0..(n >> 1) {
            self = self.swap(targets[i], targets[n - 1 - i]);
        }
        self
    }

    // Label
    pub fn label(mut self, label: String) -> Self {
        let pc = self.instructions.len();

        if let Some(idx) = self.labels.get(&label) {
            panic!("Label '{label}' was already defined on instruction row {idx}")
        }

        self.labels.insert(label, pc);

        // After a new label has been added we try to resolve unresolved labels and patch instructions
        self.try_to_patch_instructions();
        self
    }

    fn try_to_resolve_label(&mut self, label: String) -> Option<usize> {
        if let Some(&pc) = self.labels.get(&label) {
            return Some(pc);
        }

        // 1. If label doesnt exist, add it to list of unresolved labels with accompanying instruction index
        let pair = (label, self.instructions.len());
        self.unresolved_labels.push(pair);
        None
    }

    fn try_to_patch_instructions(&mut self) {
        let mut to_remove = Vec::new();

        // 2. Patch instructions
        for (label, pc) in self.unresolved_labels.clone() {
            let Some(resolved_pc) = self.try_to_resolve_label(label.clone()) else {
                continue;
            };

            let inst = &mut self.instructions[pc];

            match inst {
                Instruction::Jump(jump_pc) => *jump_pc = resolved_pc,
                Instruction::JumpIf(_expr, jump_pc) => *jump_pc = resolved_pc,
                _ => continue,
            };

            to_remove.push((label.clone(), pc));
        }

        // 3. Remove resolved labels after patching
        self.unresolved_labels
            .retain(|(label, idx)| !to_remove.contains(&(label.clone(), *idx)));
    }

    /// Inverts a non-hybrid circuit.
    pub fn inverse(&self) -> Self {
        let mut inverted_circuit = Circuit::new(self.n_qubits());
        for instruction in &self.instructions {
            match instruction {
                Instruction::Gate(gate) => inverted_circuit
                    .instructions
                    .push(Instruction::Gate(gate.inverse())),
                _ => panic!("Circuit is hybrid"),
            }
        }
        inverted_circuit.instructions.reverse();
        inverted_circuit
    }

    // Breakpoint

    pub fn breakpoint(mut self) -> Self {
        self.breakpoints.insert_or_enable(self.instructions.len());
        self
    }

    pub fn next_break(&self, pc: &CircuitPc) -> Option<(CircuitPc, bool)> {
        let brk = self.breakpoints.next_break(pc.pc())?;
        Some((CircuitPc::new(brk.pc()), brk.enabled()))
    }

    pub fn next_enabled_break(&self, pc: &CircuitPc) -> Option<CircuitPc> {
        while let Some((pc, enabled)) = self.next_break(pc) {
            if enabled {
                return Some(pc);
            }
        }
        None
    }

    pub fn breakpoint_at(&self, pc: &CircuitPc) -> Option<&Breakpoint> {
        self.breakpoints.get(pc.pc())
    }

    pub fn enabled_breakpoint_at(&self, pc: &CircuitPc) -> bool {
        self.breakpoint_at(pc)
            .map(Breakpoint::enabled)
            .unwrap_or(false)
    }

    pub fn insert_breakpoint(&mut self, pc: &CircuitPc) -> IEBreakpoint {
        self.breakpoints.insert_or_enable(pc.pc())
    }

    pub fn enable_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        self.breakpoints.enable(pc.pc())
    }

    pub fn disable_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        self.breakpoints.disable(pc.pc())
    }

    pub fn delete_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        self.breakpoints.delete(pc.pc())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cart,
        circuit::Circuit,
        ext::expand_matrix_from_gate,
        instruction::Instruction,
        simulator::{BuildSimulator, RunnableSimulator},
        sv_simulator::SVSimulator,
    };
    use nalgebra::{Complex, DMatrix, DVector, dvector};

    fn is_matrix_equal_to(m1: DMatrix<Complex<f64>>, m2: DMatrix<Complex<f64>>) -> bool {
        m1.iter()
            .zip(m2.iter())
            .all(|(a, b)| nalgebra::ComplexField::abs(a - b) < 0.001)
    }

    macro_rules! assert_is_matrix_equal {
        ($m1: expr, $m2: expr) => {
            assert!(is_matrix_equal_to($m1, $m2))
        };
    }

    fn is_vector_equal_to(v1: DVector<Complex<f64>>, v2: DVector<Complex<f64>>) -> bool {
        let l = v1.len();
        let m1 = DMatrix::<Complex<f64>>::from_row_slice(l, 1, v1.as_slice());
        let m2 = DMatrix::<Complex<f64>>::from_row_slice(l, 1, v2.as_slice());
        l == v2.len() && is_matrix_equal_to(m1, m2)
    }
    macro_rules! assert_is_vector_equal {
        ($m1: expr, $m2: expr) => {
            assert!(is_vector_equal_to($m1, $m2))
        };
    }
    fn concat_circuits(circuit1: &Circuit, circuit2: &Circuit) -> Circuit {
        let mut circuit_tot = Circuit::new(std::cmp::max(circuit1.n_qubits(), circuit2.n_qubits()));
        circuit_tot.instructions =
            [circuit1.instructions.clone(), circuit2.instructions.clone()].concat();
        circuit_tot
    }
    #[test]
    fn inverse_test() {
        let circ = Circuit::new(5)
            .h(0)
            .h(1)
            .h(3)
            .x(0)
            .y(1)
            .z(2)
            .s(4)
            .cx(&[0], 1)
            .cx(&[4], 1)
            .u(23.3, 34.5, 56.1, 0)
            .cu(1.0, 22.2, 0.1, &[4], 2)
            .swap(3, 4)
            .cswap(&[0], 1, 2);
        let circ_and_inv = concat_circuits(&circ, &circ.inverse());
        let dim = 1 << 5;
        let id = DMatrix::<Complex<f64>>::identity(dim, dim);
        let mut res: DMatrix<Complex<f64>> = id.clone();
        for instruction in circ_and_inv.instructions() {
            match instruction {
                Instruction::Gate(gate) => res = expand_matrix_from_gate(gate, 5) * res,
                _ => panic!("circ should be non-hybrid"),
            }
        }
        assert_is_matrix_equal!(id, res);
    }
    #[test]
    fn qft_test() {
        let sim =
            SVSimulator::build(Circuit::new(4).x(0).y(1).z(2).h(3).qft(&[0, 1, 2, 3])).unwrap();

        let expected_vec = dvector![
            cart!(0.0, 0.35355),  // |0000>
            cart!(0.0),           // |0001>
            cart!(-0.25, -0.25),  // |0010>
            cart!(0.0),           // |0011>
            cart!(0.35355, 0.0),  // |0100>
            cart!(0.0),           // |0101>
            cart!(-0.25, 0.25),   // |0110>
            cart!(0.0),           // |0111>
            cart!(0.0, -0.35355), // |1000>
            cart!(0.0),           // |1001>
            cart!(0.25, 0.25),    // |1010>
            cart!(0.0),           // |1011>
            cart!(-0.35355, 0.0), // |1100>
            cart!(0.0),           // |1101>
            cart!(0.25, -0.25),   // |1110>
            cart!(0.0),           // |1111>
        ];
        assert_is_vector_equal!(expected_vec, sim.final_state());
    }
}
