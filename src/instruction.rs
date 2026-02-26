use crate::{expr_dsl::Expr, gate::{Gate, QBits}};

#[derive(Debug, Clone)]
pub enum Instruction {
    Gate(Gate),
    Measurement(QBits, String),
    Jump(usize),
    JumpIf(Expr, usize),
    Assign(Expr, String),
}
