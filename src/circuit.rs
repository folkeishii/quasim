use crate::Instruction;

pub struct Circuit {
    instructions: Vec<Instruction>,
    n_qbits: usize,
}
