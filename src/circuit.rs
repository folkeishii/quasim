use std::{slice, vec};

use crate::Instruction;

pub struct Circuit {
    pub instructions: Vec<Instruction>,
    pub n_qubits: usize,
}
