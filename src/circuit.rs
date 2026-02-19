use crate::instruction::Instruction;

#[derive(Debug, Clone)]
pub struct Circuit {
    instructions: Vec<Instruction>,
    n_qubits: usize,
}

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Circuit {
            instructions: Vec::<Instruction>::default(),
            n_qubits,
        }
    }

    pub fn from_instructions<I: Into<Vec<Instruction>>>(n_qubits: usize, instructions: I) -> Self {
        Self {
            instructions: instructions.into(),
            n_qubits,
        }
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn x(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::X(target));
        self
    }

    pub fn y(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Y(target));
        self
    }

    pub fn z(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Z(target));
        self
    }

    pub fn hadamard(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::H(target));
        self
    }

    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.instructions.push(Instruction::CNOT(control, target));
        self
    }
}
