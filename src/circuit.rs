use std::{fs::read_to_string, io};

use oq3_syntax::{SourceFile, SyntaxNode, SyntaxText, ast::AstNode};

use crate::instruction::Instruction;

#[derive(Debug, Clone)]
pub struct Circuit {
    pub instructions: Vec<Instruction>,
    pub n_qubits: usize,
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
