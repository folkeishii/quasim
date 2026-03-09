use crate::{expr_dsl::Expr, gate::Gate};

#[derive(Debug, Clone)]
pub enum Instruction {
    Gate(Gate),
    MeasureBit(usize, (String, usize)), // Qbit, (Creg, bit)
    MeasureAll(String),                 // Creg
    Jump(usize),
    JumpIf(Expr, usize),
    Assign(Expr, String),
}
