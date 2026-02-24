use crate::{expr_dsl::Expr, gate::{Gate, QBits}};

#[derive(Debug, Clone)]
pub enum Instruction {
    Gate(Gate),
    Measurement(QBits, usize),
    Jump(usize),
    JumpIf(Expr, usize),
    Assign(Expr, usize),
}
