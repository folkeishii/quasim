use std::{slice, vec};

use crate::Instruction;

/// # Circuit
/// Based on OpenQASM. All qubits are initialized to the zero state.
///
/// Using circuit a simulator can then be built.
/// ## Example 1
/// OpenQASM:
/// ```
/// qubit[2] qs;
/// reset qs;
/// h qs[0];
/// cx qs[0], qs[1];
/// bit[2] my_result = measure qs;
/// ```
/// Using `Circuit`:
/// ```
/// let circuit = Circuit::default()
///     .hadamard(0)
///     .cnot(0, 1)
///     .measure(None); // Unspecified targets, measures all qubits
/// ```
#[derive(Debug, Clone, Default)]
pub struct Circuit {
    instructions: Vec<Instruction>,
    n_qubits: usize,
}

impl Circuit {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn iter(&self) -> slice::Iter<'_, Instruction> {
        self.instructions.iter()
    }

    pub fn push_instruction(&mut self, instruction: Instruction) {
        self.extend_qubits_by_iter(instruction.target.iter());
        self.instructions.push(instruction);
    }

    pub fn x(mut self, target: usize) -> Self {
        self.extend_qubits(target);

        self.push_instruction(Instruction::x(target));

        self
    }

    pub fn y(mut self, target: usize) -> Self {
        self.extend_qubits(target);

        self.push_instruction(Instruction::y(target));

        self
    }

    pub fn z(mut self, target: usize) -> Self {
        self.extend_qubits(target);

        self.push_instruction(Instruction::z(target));

        self
    }

    pub fn t(mut self, target: usize) -> Self {
        self.extend_qubits(target);

        self.push_instruction(Instruction::t(target));

        self
    }

    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.extend_qubits(control);
        self.extend_qubits(target);

        self.push_instruction(Instruction::cnot(control, target));

        self
    }

    pub fn hadamard(mut self, target: usize) -> Self {
        self.extend_qubits(target);

        self.push_instruction(Instruction::hadamard(target));

        self
    }

    /// If targets is `None`, measure all qubits
    pub fn measure(mut self, targets: Option<Vec<usize>>) -> Self {
        if let Some(targets) = targets {
            self.extend_qubits_by_iter(targets.iter());
        }

        todo!(); // Extend `gates` with the hadamard gate

        self
    }

    fn extend_qubits(&mut self, include: usize) {
        self.n_qubits = self.n_qubits.max(include + 1)
    }

    fn extend_qubits_by_iter<'a, I>(&mut self, include: I)
    where
        I: Iterator<Item = &'a usize>,
    {
        if let Some(mx) = include.max() {
            self.n_qubits = *mx + 1;
        }
    }
}

impl IntoIterator for Circuit {
    type Item = Instruction;
    type IntoIter = vec::IntoIter<Instruction>;

    fn into_iter(self) -> Self::IntoIter {
        self.instructions.into_iter()
    }
}
