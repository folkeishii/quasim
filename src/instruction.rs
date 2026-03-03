use crate::{
<<<<<<< HEAD
    circuit::LabelPc,
    expr_dsl::Expr,
    gate::{Gate, QBits},
=======
    circuit::CircuitPc, expr_dsl::Expr, gate::{Gate, QBits}
>>>>>>> 97d1bf4 (CircuitLabel)
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
