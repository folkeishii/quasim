use std::{
    collections::{HashMap, HashSet},
    f64::consts::PI,
    fmt::{Display, Write},
};

use crate::{
    expr_dsl::Expr,
    gate::{Gate, GateType, QBits},
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
    registers: HashSet<String>,
    labels: HashMap<String, LabelPc>,
    unresolved_labels: Vec<(String, LabelPc)>,
    sub_circuits: HashMap<String, SubCircuit>,
    unresolved_sub_circuits: Vec<(String, LabelPc)>,
}

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            instructions: Vec::<Instruction>::default(),
            n_qubits: n_qubits,
            registers: HashSet::new(),
            labels: HashMap::new(),
            unresolved_labels: Vec::new(),
            sub_circuits: HashMap::new(),
            unresolved_sub_circuits: Vec::new(),
        }
    }

    pub fn new_reg<I: Into<String>>(mut self, name: I) -> Self {
        self.registers.insert(name.into());
        self
    }

    pub fn new_sub_circuit<I: Into<String>>(
        mut self,
        name: I,
        circuit: Circuit,
    ) -> Self {
        let name = name.into();

        let (
            sub_circuit,
            SubCircuitPeriphs {
                registers,
                labels,
                unresolved_labels,
                sub_circuits,
                unresolved_sub_circuits,
            },
        ) = SubCircuit::from_circuit(circuit);

        if let Some(existing) = self.sub_circuits.get(&name)
            && sub_circuit.eq(existing)
        {
            panic!("Cannot add {}: A different sub circuit with the same name already exists", name)
        }

        for (imported_name, imported_sub_circuit) in sub_circuits.iter() {
            if let Some(existing) = self.sub_circuits.get(imported_name)
                && !imported_sub_circuit.eq(existing)
            {
                panic!("Cannot add {}: Duplicate definition of sub circuit {}", name, imported_name)
            } else if name.eq(imported_name) && !sub_circuit.eq(imported_sub_circuit) {
                panic!("Cannot add {}: Duplicate definition of sub circuit {}", name, imported_name)
            }
        }

        self.registers.extend(registers);
        self.labels.extend(
            labels
                .into_iter()
                .map(|(label, label_pc)| (label, label_pc.map_sub_circuit(name.clone()))),
        );
        self.unresolved_labels.extend(
            unresolved_labels
                .into_iter()
                .map(|(label, label_pc)| (label, label_pc.map_sub_circuit(name.clone()))),
        );
        self.unresolved_sub_circuits.extend(
            unresolved_sub_circuits
                .into_iter()
                .map(|(label, label_pc)| (label, label_pc.map_sub_circuit(name.clone())))
                .filter(|(label, _)| label != &name),
        );
        self.unresolved_sub_circuits.retain_mut(|(unresolved_name, _)| unresolved_name != &name);
        self.sub_circuits.insert(name, sub_circuit);
        self
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

    // Sub circuit

    pub fn sub_circuit<I: Into<String>>(mut self, name: I, first_qubit: usize) -> Self {
        self.instructions.push(Instruction::SubCircuit(name.into(), first_qubit));
        self
    }

    // Classical instructions

    pub fn measure_bit(mut self, target: usize, reg: &str) -> Self {
        self.instructions.push(Instruction::Measurement(
            QBits::from_bitstring(1 << target),
            reg.to_owned(),
        ));
        self
    }

    pub fn measure(mut self, targets: &[usize], reg: &str) -> Self {
        self.instructions.push(Instruction::Measurement(
            QBits::from_indices(targets),
            reg.to_owned(),
        ));
        self
    }

    pub fn jump(mut self, label: &str) -> Self {
        let label_pc = match self.try_to_resolve_label(label) {
            Some(label_pc) => label_pc.clone(),
            None => LabelPc::at_main(0), // Placeholder pc
        };

        self.instructions.push(Instruction::Jump(label_pc));
        self
    }

    pub fn jump_if(mut self, expr: Expr, label: &str) -> Self {
        let label_pc = match self.try_to_resolve_label(label) {
            Some(label_pc) => label_pc.clone(),
            None => LabelPc::at_main(0), // Placeholder pc
        };

        self.instructions.push(Instruction::JumpIf(expr, label_pc));
        self
    }

    /// Conditionally apply whichever instruction that comes after
    pub fn apply_if(mut self, expr: Expr) -> Self {
        self.instructions.push(Instruction::JumpIf(
            !expr,
            LabelPc::at_main(self.instructions.len() + 2),
        ));
        self
    }

    pub fn reset(self, target: usize) -> Self {
        self.new_reg("_reset")
            .measure_bit(target, "_reset")
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
    pub fn label(mut self, label: &str) -> Self {
        let pc = self.instructions.len();

        if let Some(idx) = self.labels.get(label) {
            panic!("Label '{label}' was already defined on instruction row {idx}")
        }

        self.labels.insert(label.to_owned(), LabelPc::at_main(pc));

        // After a new label has been added we try to resolve unresolved labels and patch instructions
        self.try_to_patch_instructions();
        self
    }

    fn try_to_resolve_label(&mut self, label: &str) -> Option<&LabelPc> {
        if let Some(label_pc) = self.labels.get(label) {
            return Some(label_pc);
        }

        // 1. If label doesnt exist, add it to list of unresolved labels with accompanying instruction index
        let pair = (label.to_owned(), LabelPc::at_main(self.instructions.len()));
        self.unresolved_labels.push(pair);
        None
    }

    fn try_to_patch_instructions(&mut self) {
        let mut to_remove = Vec::new();

        // 2. Patch instructions
        for (label, label_pc) in self.unresolved_labels.clone() {
            let inst_index = label_pc.pc();

            if let Some(_sub_circuit) = label_pc.sub_circuit() {
                todo!("TODO: handle label inside sub circuit")
            }

            let Some(resolved_pc) = self.try_to_resolve_label(&label).cloned() else {
                continue;
            };

            let inst = &mut self.instructions[inst_index];

            match inst {
                Instruction::Jump(jump_label_pc) => *jump_label_pc = resolved_pc,

                Instruction::JumpIf(_expr, jump_label_pc) => {
                    *jump_label_pc = resolved_pc;
                }

                _ => continue,
            };

            to_remove.push((label.clone(), label_pc));
        }

        // 3. Remove resolved labels after patching
        self.unresolved_labels
            .retain(|(label, idx)| !to_remove.contains(&(label.clone(), idx.clone())));
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubCircuit {
    instructions: Vec<Instruction>,
    n_qubits: usize,
}
impl SubCircuit {
    pub fn from_circuit(circuit: Circuit) -> (Self, SubCircuitPeriphs) {
        let Circuit {
            instructions,
            n_qubits,
            registers,
            labels,
            unresolved_labels,
            sub_circuits,
            unresolved_sub_circuits
        } = circuit;
        (
            Self {
                instructions,
                n_qubits,
            },
            SubCircuitPeriphs {
                registers,
                labels,
                unresolved_labels,
                sub_circuits,
                unresolved_sub_circuits
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct SubCircuitPeriphs {
    registers: HashSet<String>,
    labels: HashMap<String, LabelPc>,
    unresolved_labels: Vec<(String, LabelPc)>,
    sub_circuits: HashMap<String, SubCircuit>,
    unresolved_sub_circuits: Vec<(String, LabelPc)>
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabelPc {
    sub_circuit: Option<String>,
    pc: usize,
}
impl LabelPc {
    pub fn at_main(pc: usize) -> Self {
        LabelPc {
            sub_circuit: None,
            pc,
        }
    }

    pub fn at_sub_circuit(sub_circuit: String, pc: usize) -> Self {
        LabelPc {
            sub_circuit: Some(sub_circuit),
            pc,
        }
    }

    /// Maps a None into Some(sub_circuit)
    pub fn map_sub_circuit(self, sub_circuit: String) -> Self {
        match self {
            LabelPc {
                sub_circuit: None,
                pc,
            } => Self::at_sub_circuit(sub_circuit, pc),
            label_pc => label_pc,
        }
    }

    pub fn sub_circuit(&self) -> Option<&String> {
        self.sub_circuit.as_ref()
    }

    pub fn pc(&self) -> usize {
        self.pc
    }
}
impl Display for LabelPc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('[')?;
        if let Some(sc) = self.sub_circuit.as_ref() {
            write!(f, "{};", sc)?;
        }
        write!(f, "{}]", self.pc)
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
