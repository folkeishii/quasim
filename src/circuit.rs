use std::fs::read_to_string;

use oq3_syntax::{SourceFile, SyntaxNode, ast::AstNode, SyntaxText};

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
            VERSION_STRING => (),
            EXPR_STMT => {
                //println!("Expression statement: {:?}", child);
                recurse(&child);
            },
            GATE_CALL_EXPR => {
                Self::parse_gate_call_expr(&child);
            },
            /*
            INDEX_OPERATOR => {
                println!("Index operator: {:?}", child);
                recurse(&child);
            },
            EXPRESSION_LIST => {
                println!("Expression list: {:?}", child);
                recurse(&child);
            },
            */
            LITERAL => {
                println!("Literal, literally: {:?}", child.text());
                recurse(&child);
            }
            _ => {
                // println!("Unknown instruction: {:?}", child),
            }
        });
        println!(""); // Just a blank line to separate groups
        return instructions;
    }

    // This may have to return a Vec instead, depending on if OpenQASM gate calls
    // can represent multiple gates in a circuit...
    fn parse_gate_call_expr(node: &SyntaxNode) -> Result<Instruction, ()> {
        use oq3_syntax::SyntaxKind::*;
        let mess = Self::instructions_from_syntax_tree;
    
        let Some(gate_name) = Self::check_first_for_identifier(node) else {return Err(())};
        println!("Identifier found, literally: '{}'", gate_name);

        node.children().for_each(|child| match child.kind() {
            IDENTIFIER => (), // Handled seperately before this 
            QUBIT_LIST => { 
                println!("- Indexes in qubit list: {:?}", Self::indexes_in_qubit_list(&child)); 
            },
            _ => {
                //println!("Unknown instruction: {:?}", child);
                mess(&child);
            }
        });

        Ok(Instruction::X(0)) // dummy for testing, DO NOT MERGE
    }
    
    fn check_first_for_identifier(node: &SyntaxNode) -> Option<SyntaxText> {
        if let Some(ident) = node.first_child() {
            if ident.kind() == oq3_syntax::SyntaxKind::IDENTIFIER {
                return Some(ident.text());
            }
        }
        return None;
    }

    fn indexes_in_qubit_list(node: &SyntaxNode) -> Vec<usize> {
        let mut result: Vec<usize> = vec![];
        node.children().for_each(|child| match child.kind() {
            oq3_syntax::SyntaxKind::LITERAL => {
                result.push(child.text().to_string().parse::<usize>().unwrap()); // SUPER BADDDDDDDDD
            },
            _ => { result.append(&mut Self::indexes_in_qubit_list(&child)); },
        });
        return result;
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
