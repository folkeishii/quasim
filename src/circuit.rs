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
    n_qbits: usize,
}

impl Circuit {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_gate(mut self, gate: Instruction) {
        self.extend_qbits_by_iter(gate.target.iter());
        self.instructions.push(gate);
    }

    pub fn x(mut self, target: usize) -> Self {
        self.extend_qbits(target);

        todo!(); // Extend `gates` with the X gate
        // ex. self.push_gate(Gate::x(target))

        self
    }

    pub fn y(mut self, target: usize) -> Self {
        self.extend_qbits(target);

        todo!(); // Extend `gates` with the Y gate

        self
    }

    pub fn z(mut self, target: usize) -> Self {
        self.extend_qbits(target);

        todo!(); // Extend `gates` with the Z gate

        self
    }

    pub fn t(mut self, target: usize) -> Self {
        self.extend_qbits(target);

        todo!(); // Extend `gates` with the T gate

        self
    }

    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.extend_qbits(control);
        self.extend_qbits(target);

        todo!(); // Extend `gates` with the cnot gate
        // ex. self.push_gate(Gate::cnot(control, target))

        self
    }

    pub fn hadamard(mut self, target: usize) -> Self {
        self.extend_qbits(target);

        todo!(); // Extend `gates` with the hadamard gate

        self
    }

    /// If targets is `None`, measure all qubits
    pub fn measure(mut self, targets: Option<Vec<usize>>) -> Self {
        if let Some(targets) = targets {
            self.extend_qbits_by_iter(targets.iter());
        }

        todo!(); // Extend `gates` with the hadamard gate

        self
    }

    fn extend_qbits(&mut self, include: usize) {
        self.n_qbits = self.n_qbits.max(include + 1)
    }

    fn extend_qbits_by_iter<'a, I>(&mut self, include: I)
    where
        I: Iterator<Item = &'a usize>,
    {
        if let Some(mx) = include.max() {
            self.n_qbits = *mx + 1;
        }
    }
}
