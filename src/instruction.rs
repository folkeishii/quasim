use crate::{gate::{Gate, QBits}};

#[derive(Debug, Clone)]
pub enum Instruction {
    Gate(Gate),
    Measurement(QBits),
}
