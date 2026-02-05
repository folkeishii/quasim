use std::{slice, vec};

use crate::Instruction;

pub struct Circuit {
    instructions: Vec<Instruction>,
    n_qubits: usize,
}
