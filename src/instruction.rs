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
    Call(String, usize),
}
impl From<PureInstruction> for Instruction {
    fn from(value: PureInstruction) -> Self {
        match value {
            PureInstruction::Gate(gate) => Self::Gate(gate),
            PureInstruction::Call(name, lsq) => Self::Call(name, lsq),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PureInstruction {
    Gate(Gate),
    /// `Call(name, lsq)`
    Call(String, usize),
}
impl From<Gate> for PureInstruction {
    fn from(value: Gate) -> Self {
        Self::Gate(value)
    }
}
