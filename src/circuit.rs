use std::collections::{HashMap, HashSet};

use crate::{
    expr_dsl::Expr,
    gate::{Gate, GateType, QBits},
    instruction::Instruction,
};
use std::{fs::read_to_string, io};
use oq3_syntax::{SourceFile, SyntaxNode, SyntaxText, ast::AstNode};

#[derive(Debug, Clone)]
pub struct Circuit {
    instructions: Vec<Instruction>,
    n_qubits: usize,
    registers: HashSet<String>,
    labels: HashMap<String, usize>,
    unresolved_labels: Vec<(String, usize)>,
}

#[derive(Debug, thiserror::Error)]
pub enum QASMParseError {
    #[error("could not read QASM source file: {0}")]
    FileError(#[from] io::Error),
    #[error("a gate call was missing qubits")]
    GateCallMissingQubits,
    #[error("gate {0} was called with {1} qubits, but it requires {2}")]
    WrongNumberOfQubits(String, usize, usize), // gate name, number of qubits provided, number of qubits required
    #[error("unrecognized gate: {0}")]
    UnrecognizedGate(String),
    #[error("gate call was missing the name of the gate to call, how even..?")]
    UnlabeledGateCall,
    #[error("tried to parse '{0}' as a qubit index, but it wasn't a valid number")]
    FailedToParseQubitIndex(String),
}

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            instructions: Vec::<Instruction>::default(),
            n_qubits: n_qubits,
            registers: HashSet::new(),
            labels: HashMap::new(),
            unresolved_labels: Vec::new(),
        }
    }

    pub fn new_reg(mut self, name: &str) -> Self {
        self.registers.insert(name.to_owned());
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

    fn print_tree(file: SourceFile) {
        for item in file.syntax().descendants() {
            println!("{:?}", item);
        }
    }

    fn print_children(file: SourceFile) {
        for item in file.syntax().children() {
            println!("{:?}", item);
        }
    }

    pub fn from_qasm_file(file_name: &str) -> Result<Self, QASMParseError> {
        let file_string = read_to_string(file_name)?;
        let parsed_source = SourceFile::parse(&file_string);
        let parse_tree: SourceFile = parsed_source.tree();
        println!(
            "Found {} statements",
            parse_tree.statements().collect::<Vec<_>>().len()
        );
        let syntax_errors = parsed_source.errors();
        println!(
            "Found {} parse errors:\n{:?}\n",
            syntax_errors.len(),
            syntax_errors
        );
        let instructions = Self::instructions_from_syntax_tree(parse_tree.syntax())?;
        let n_qubits = Self::number_of_qubits(&instructions);
        return Ok(Circuit {
            instructions,
            n_qubits,
        });
    }

    fn number_of_qubits(instructions: &Vec<Instruction>) -> usize {
        let mut greatest_found: usize = 0;
        for instruction in instructions {
            for index in instruction.get_controls() {
                if index > greatest_found {
                    greatest_found = index;
                }
            }
            for index in instruction.get_targets() {
                if index > greatest_found {
                    greatest_found = index;
                }
            }
        }
        return greatest_found + 1; // +1 because qubits are 0-indexed
    }

    fn instructions_from_syntax_tree(
        node: &SyntaxNode,
    ) -> Result<Vec<Instruction>, QASMParseError> {
        use oq3_syntax::SyntaxKind::*;
        let recurse = Self::instructions_from_syntax_tree;
        let mut instructions = Vec::<Instruction>::default();
        for child in node.children() {
            match child.kind() {
                EXPR_STMT => {
                    // Expression statements can contain gate calls!
                    instructions.append(&mut recurse(&child)?);
                }
                GATE_CALL_EXPR => {
                    instructions.push(Self::parse_gate_call_expr(&child)?);
                }
                LITERAL => {
                    // A literal has no children
                    println!("Literal, literally: {:?}", child.text());
                }
                _ => {
                    // Ignore everything else at top level for now... Huge TODO
                    // println!("Unknown instruction: {:?}", child),
                }
            }
        }
        return Ok(instructions);
    }

    // This may have to return a Vec instead, depending on if OpenQASM gate calls
    // can represent multiple gates in a circuit...
    fn parse_gate_call_expr(node: &SyntaxNode) -> Result<Instruction, QASMParseError> {
        use oq3_syntax::SyntaxKind::*;

        let mut gate_name: Option<SyntaxText> = None;
        let mut qubit_indexes: Vec<usize> = vec![];

        for child in node.children() {
            match child.kind() {
                IDENTIFIER => gate_name = Some(child.text()),
                QUBIT_LIST => {
                    qubit_indexes = Self::indexes_in_qubit_list(&child)?;
                }
                _ => (),
            }
        }

        if let Some(name) = gate_name {
            if qubit_indexes.len() == 0 {
                return Err(QASMParseError::GateCallMissingQubits);
            }

            // TODO: Is there a more "automatic" way to do this?
            match name.to_string().as_str() {
                "x" => return Ok(Instruction::X(qubit_indexes[0])),
                "y" => return Ok(Instruction::Y(qubit_indexes[0])),
                "z" => return Ok(Instruction::Z(qubit_indexes[0])),
                "h" => return Ok(Instruction::H(qubit_indexes[0])),
                "cx" => {
                    if qubit_indexes.len() != 2 {
                        return Err(QASMParseError::WrongNumberOfQubits(
                            name.to_string(),
                            qubit_indexes.len(),
                            2,
                        ));
                    }
                    return Ok(Instruction::CNOT(qubit_indexes[0], qubit_indexes[1]));
                }
                _ => return Err(QASMParseError::UnrecognizedGate(name.to_string())),
            }
        } else {
            return Err(QASMParseError::UnlabeledGateCall);
        }
    }

    fn find_first_identifier(node: &SyntaxNode) -> Option<SyntaxText> {
        for child in node.children() {
            if child.kind() == oq3_syntax::SyntaxKind::IDENTIFIER {
                return Some(child.text());
            } else {
                if let Some(ident) = Self::find_first_identifier(&child) {
                    return Some(ident);
                }
            }
        }
        return None;
    }

    fn indexes_in_qubit_list(node: &SyntaxNode) -> Result<Vec<usize>, QASMParseError> {
        let mut result: Vec<usize> = vec![];

        for child in node.children() {
            match child.kind() {
                oq3_syntax::SyntaxKind::LITERAL => {
                    let parse_attempt = child.text().to_string().parse::<usize>();
                    if let Ok(index) = parse_attempt {
                        result.push(index);
                    } else {
                        return Err(QASMParseError::FailedToParseQubitIndex(
                            child.text().to_string(),
                        ));
                    }
                }
                _ => {
                    result.append(&mut Self::indexes_in_qubit_list(&child)?);
                }
            }
        }
        return Ok(result);
    }

    // Under here are all the builder functions.

    pub fn x(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::X, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn y(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Y, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn z(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Z, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn hadamard(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::H, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::X, &[control], &[target]).unwrap(),
        ));
        self
    }

    pub fn swap(mut self, target1: usize, target2: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::SWAP, &[], &[target1, target2]).unwrap(),
        ));
        self
    }

    pub fn fredkin(mut self, control: usize, target1: usize, target2: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::SWAP, &[control], &[target1, target2]).unwrap(),
        ));
        self
    }

    pub fn u(mut self, theta: f64, phi: f64, lambda: f64, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, phi, lambda), &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cu(mut self, theta: f64, phi: f64, lambda: f64, control: usize, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, phi, lambda), &[control], &[target]).unwrap(),
        ));
        self
    }

    pub fn s(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::S, &[], &[target]).unwrap(),
        ));
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

    pub fn jump(mut self, label: &str) -> Self {
        let pc = match self.try_to_resolve_label(label) {
            Some(label_pc) => label_pc,
            None => 0, // Placeholder pc
        };

        self.instructions.push(Instruction::Jump(pc));
        self
    }

    pub fn jump_if(mut self, expr: Expr, label: &str) -> Self {
        let pc = match self.try_to_resolve_label(label) {
            Some(label_pc) => label_pc,
            None => 0, // Placeholder pc
        };

        self.instructions.push(Instruction::JumpIf(expr, pc));
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

    // Label

    pub fn label(mut self, label: &str) -> Self {
        let pc = self.instructions.len();

        if let Some(idx) = self.labels.get(label) {
            panic!("Label '{label}' was already defined on instruction row {idx}")
        }

        self.labels.insert(label.to_owned(), pc);

        // After a new label has been added we try to resolve unresolved labels and patch instructions
        self.try_to_patch_instructions();
        self
    }

    fn try_to_resolve_label(&mut self, label: &str) -> Option<usize> {
        if let Some(pc) = self.labels.get(label) {
            return Some(*pc);
        }

        // 1. If label doesnt exist, add it to list of unresolved labels with accompanying instruction index
        let pair = (label.to_owned(), self.instructions.len());
        self.unresolved_labels.push(pair);
        None
    }

    fn try_to_patch_instructions(&mut self) {
        let mut to_remove = Vec::new();

        // 2. Patch instructions
        for (label, inst_index) in &self.unresolved_labels.clone() {
            if let Some(resolved_pc) = self.try_to_resolve_label(label) {
                let inst = &mut self.instructions[*inst_index];

                *inst = match inst {
                    Instruction::Jump(_) => Instruction::Jump(resolved_pc),

                    Instruction::JumpIf(expr, _) => Instruction::JumpIf(expr.clone(), resolved_pc),

                    _ => continue,
                };

                to_remove.push((label.clone(), *inst_index));
            }
        }

        // 3. Remove resolved labels after patching
        self.unresolved_labels
            .retain(|(label, idx)| !to_remove.contains(&(label.clone(), *idx)));
    }
}
