use crate::{
    circuit::pc::CircuitPc, expr_dsl::Expr, gate::{Gate, QBits}
};

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    Gate(Gate),
    Measurement(QBits, String),
    Jump(CircuitPc),
    JumpIf(Expr, CircuitPc),
    Assign(Expr, String),
    SubCircuit(String, usize) // Sub circuit name, first qubit
}
