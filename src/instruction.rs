use crate::{
    expr_dsl::Expr,
    gate::{Gate, QBits},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    Gate(Gate),
    Measurement(QBits, String), // Targets, Register, Register bit offset
    Jump(usize),                // Cannot jump into another sub circuit
    JumpIf(Expr, usize),        // Cannot jump into another sub circuit
    Assign(Expr, String),
}
