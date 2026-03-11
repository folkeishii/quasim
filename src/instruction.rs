use crate::{expr_dsl::Expr, gate::Gate};

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    Gate(Gate),
    /// `MeasureBit(Qbit, (Creg, bit))`
    MeasureBit(usize, (String, usize)),
    /// `MeasureAll(Creg)`
    MeasureAll(String),
    Jump(usize),
    JumpIf(Expr, usize),
    Assign(Expr, String),
    /// `Call(name, lsq)`
    Call(String, usize)
}
