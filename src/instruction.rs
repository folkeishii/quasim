use crate::{
    expr_dsl::Expr,
    gate::{Gate, QBits},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    Gate(Gate),
    Measurement(QBits, String),
    Jump(usize),         // Cannot jump into another sub circuit
    JumpIf(Expr, usize), // Cannot jump into another sub circuit
    Assign(Expr, String),
    SubCircuit(String, usize), // Sub circuit name, least significant qubit inside sub circuit
}
