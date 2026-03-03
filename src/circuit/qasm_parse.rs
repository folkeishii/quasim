use std::io;

use log::{info, warn};
use oq3_syntax::{SyntaxNode, SyntaxText};

use crate::circuit::Circuit;

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

/// This does not work well if there are multiple qubit registers.
/// All this does is find the greatest qubit index used in the program,
/// and adds 1 to it to get the number of qubits...
/// TODO: make this work with multiple registers, be wary of register offsets (?)
pub fn count_qubits_from_syntax_tree(node: &SyntaxNode) -> Result<usize, QASMParseError> {
    use oq3_syntax::SyntaxKind::*;
    let mut greatest_found: usize = 0;

    for child in node.children() {
        match child.kind() {
            EXPR_STMT => {
                // Expression statements can contain gate calls!
                let count = count_qubits_from_syntax_tree(&child)?;
                if count > greatest_found {
                    greatest_found = count;
                }
            }
            GATE_CALL_EXPR => {
                let qubits = extract_qubits_from_gate_call(&child)?;
                for qubit in qubits {
                    if qubit > greatest_found {
                        greatest_found = qubit;
                    }
                }
            }
            _ => {}
        }
    }

    return Ok(greatest_found + 1); // +1 because qubits are 0-indexed
}

/// Extracts qubit indexes from a gate call node.
fn extract_qubits_from_gate_call(node: &SyntaxNode) -> Result<Vec<usize>, QASMParseError> {
    use oq3_syntax::SyntaxKind::*;
    let mut qubit_indexes: Vec<usize> = vec![];

    // TODO: Double check that a gate call was actually supplied...

    for child in node.children() {
        match child.kind() {
            QUBIT_LIST => {
                qubit_indexes = indexes_in_qubit_list(&child)?;
            }
            _ => (),
        }
    }

    Ok(qubit_indexes)
}

/// This finds gate calls in the syntax tree and applies them to the circuit by calling
/// apply_gate_call_expr on them.
pub fn apply_gates_from_syntax_tree(
    circuit: Circuit,
    node: &SyntaxNode,
) -> Result<Circuit, QASMParseError> {
    use oq3_syntax::SyntaxKind::*;
    let mut result = circuit;

    for child in node.children() {
        match child.kind() {
            VERSION_STRING => {
                info!("Found version string in QASM source: {}", child.text());
            }
            EXPR_STMT => {
                // Expression statements can contain gate calls!
                result = apply_gates_from_syntax_tree(result, &child)?;
            }
            GATE_CALL_EXPR => {
                result = apply_gate_call_expr(result, &child)?;
            }
            _ => {
                warn!("Unhandled syntax node: {:?}", child.kind());
            }
        }
    }

    return Ok(result);
}

/// This applies a gate call expression to the circuit by calling the appropriate builder method.
fn apply_gate_call_expr(
    mut circuit: Circuit,
    node: &SyntaxNode,
) -> Result<Circuit, QASMParseError> {
    use oq3_syntax::SyntaxKind::*;

    let mut gate_name: Option<SyntaxText> = None;
    let qubit_indexes: Vec<usize> = extract_qubits_from_gate_call(node)?;

    for child in node.children() {
        match child.kind() {
            IDENTIFIER => gate_name = Some(child.text()),
            _ => (),
        }
    }

    if let Some(name) = gate_name {
        if qubit_indexes.len() == 0 {
            return Err(QASMParseError::GateCallMissingQubits);
        }

        // Apply the gate via builder method
        match name.to_string().as_str() {
            "x" => circuit = circuit.x(qubit_indexes[0]),
            "y" => circuit = circuit.y(qubit_indexes[0]),
            "z" => circuit = circuit.z(qubit_indexes[0]),
            "h" => circuit = circuit.hadamard(qubit_indexes[0]),
            "cx" => {
                if qubit_indexes.len() != 2 {
                    return Err(QASMParseError::WrongNumberOfQubits(
                        name.to_string(),
                        qubit_indexes.len(),
                        2,
                    ));
                }
                circuit = circuit.cnot(qubit_indexes[0], qubit_indexes[1]);
            }
            _ => return Err(QASMParseError::UnrecognizedGate(name.to_string())),
        }
    } else {
        return Err(QASMParseError::UnlabeledGateCall);
    }

    Ok(circuit)
}

fn find_first_identifier(node: &SyntaxNode) -> Option<SyntaxText> {
    for child in node.children() {
        if child.kind() == oq3_syntax::SyntaxKind::IDENTIFIER {
            return Some(child.text());
        } else {
            if let Some(ident) = find_first_identifier(&child) {
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
                result.append(&mut indexes_in_qubit_list(&child)?);
            }
        }
    }
    return Ok(result);
}
