use std::fs::read_to_string;

use oq3_syntax::{SourceFile, SyntaxNode, SyntaxText, ast::AstNode};

use crate::instruction::Instruction;

#[derive(Debug, Clone)]
pub struct Circuit {
    pub instructions: Vec<Instruction>,
    pub n_qubits: usize,
}

// TODO: Make this a proper error type
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
        let instructions = Self::instructions_from_syntax_tree(parse_tree.syntax())?;
        return Ok(Circuit {
            instructions,
            n_qubits: 5,
        }); // dummy n_qubits for testing, DO NOT MERGE
    }

    fn instructions_from_syntax_tree(
        node: &SyntaxNode,
    ) -> Result<Vec<Instruction>, QuasimParseError> {
        use oq3_syntax::SyntaxKind::*;
        let recurse = Self::instructions_from_syntax_tree;
        let mut instructions = Vec::<Instruction>::default();
        for child in node.children() {
            match child.kind() {
                VERSION_STRING => (),
                EXPR_STMT => {
                    //println!("Expression statement: {:?}", child);
                    recurse(&child)?;
                }
                GATE_CALL_EXPR => {
                    instructions.push(Self::parse_gate_call_expr(&child)?);
                }
                LITERAL => {
                    println!("Literal, literally: {:?}", child.text());
                    recurse(&child)?;
                }
                _ => {
                    // println!("Unknown instruction: {:?}", child),
                }
            }
        }
        println!(""); // Just a blank line to separate groups
        return Ok(instructions);
    }

    // This may have to return a Vec instead, depending on if OpenQASM gate calls
    // can represent multiple gates in a circuit...
    fn parse_gate_call_expr(node: &SyntaxNode) -> Result<Instruction, QuasimParseError> {
        use oq3_syntax::SyntaxKind::*;

        let mut gate_name: Option<SyntaxText> = None;
        let mut qubit_indexes: Vec<usize> = vec![];

        for child in node.children() {
            match child.kind() {
                IDENTIFIER => gate_name = Some(child.text()),
                QUBIT_LIST => {
                    qubit_indexes = Self::indexes_in_qubit_list(&child)?;
                    println!("Qubit list found, indexes: {:?}", qubit_indexes);
                }
                _ => (),
            }
        }

        if let Some(name) = gate_name {
            println!("Gate call name: '{}'", name);

            if qubit_indexes.len() == 0 {
                return Err("Failed to find any qubits in gate call".into());
            }
        } else {
            return Err("Failed to find gate name".into());
        }

        Ok(Instruction::X(0)) // dummy for testing, DO NOT MERGE
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

    fn indexes_in_qubit_list(node: &SyntaxNode) -> Result<Vec<usize>, QuasimParseError> {
        let mut result: Vec<usize> = vec![];

        for child in node.children() {
            match child.kind() {
                oq3_syntax::SyntaxKind::LITERAL => {
                    let parse_attempt = child.text().to_string().parse::<usize>();
                    if let Ok(index) = parse_attempt {
                        result.push(index);
                    } else {
                        return Err("Failed to parse qubit index".into());
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
