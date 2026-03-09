use crate::{
    expr_dsl::Expr,
    gate::{Gate, QBits},
};

#[derive(Debug, Clone)]
pub enum Instruction {
    Gate(Gate),
    Measurement(usize, (String, usize)), // Qbit, (Creg, bit)
    Jump(usize),
    JumpIf(Expr, usize),
    Assign(Expr, String),
}
