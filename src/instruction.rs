use crate::{expr_dsl::Expr, gate::{Gate, QBits}};

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
    /// `Call(name, lsq, ctrl)`
    Call(String, usize, QBits),
}
impl From<PureInstruction> for Instruction {
    fn from(value: PureInstruction) -> Self {
        match value {
            PureInstruction::Gate(gate) => Self::Gate(gate),
            PureInstruction::Call(name, lsq, ctrl) => Self::Call(name, lsq, ctrl),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PureInstruction {
    Gate(Gate),
    /// `Call(name, lsq, ctrl)`
    Call(String, usize, QBits),
}
impl From<Gate> for PureInstruction {
    fn from(value: Gate) -> Self {
        Self::Gate(value)
    }
}
