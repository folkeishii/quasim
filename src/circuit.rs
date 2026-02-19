use std::fs::read_to_string;

use oq3_syntax::{SourceFile, SyntaxNode, ast::AstNode};

use crate::instruction::Instruction;

#[derive(Debug, Clone)]
pub struct Circuit {
    pub instructions: Vec<Instruction>,
    pub n_qubits: usize,
}

type QuasimParseError = String;

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Circuit {
            instructions: Vec::<Instruction>::default(),
            n_qubits,
        }
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

    pub fn from_qasm_file(file_name: &str) -> Result<Self, QuasimParseError> {
        let file_string = read_to_string(file_name).map_err(|e| e.to_string())?;
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
        Self::instructions_from_syntax_tree(parse_tree.syntax());
        // Self::print_children(parse_tree);
        return Ok(Circuit::new(2));
    }

    fn instructions_from_syntax_tree(node: &SyntaxNode) -> Vec<Instruction> {
        use oq3_syntax::SyntaxKind::*;
        let recurse = Self::instructions_from_syntax_tree;
        let mut instructions = Vec::<Instruction>::default();
        node.children().for_each(|child| match child.kind() {
            EXPR_STMT => {
                println!("Expression statement: {:?}", child);
                recurse(&child);
            }
            _ => println!("Unknown instruction: {:?}", child),
        });
        return instructions;
    }

    pub fn x(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::X(target));
        self
    }

    pub fn y(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Y(target));
        self
    }

    pub fn z(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Z(target));
        self
    }

    pub fn hadamard(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::H(target));
        self
    }

    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.instructions.push(Instruction::CNOT(control, target));
        self
    }
}
