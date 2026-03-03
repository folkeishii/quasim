use crate::{
    circuit::LabelPc,
    expr_dsl::Expr,
    gate::{Gate, QBits},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    Gate(Gate),
    Measurement(QBits, String),
    Jump(LabelPc),
    JumpIf(Expr, LabelPc),
    Assign(Expr, String),
    SubCircuit(String, usize) // Sub circuit name, first qubit
}
