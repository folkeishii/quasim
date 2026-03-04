use std::fmt::{Display, Write};

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CircuitPc {
    sub_circuit: Option<String>,
    pc: usize,
    // Least Significant Qubit
    lsq: usize,
}
impl CircuitPc {
    pub fn at_main(pc: usize) -> Self {
        CircuitPc {
            sub_circuit: None,
            pc,
            lsq: 0
        }
    }

    pub fn at_sub_circuit(sub_circuit: String, lsq: usize, pc: usize) -> Self {
        CircuitPc {
            sub_circuit: Some(sub_circuit),
            pc,
            lsq,
        }
    }

    /// Maps a None into Some(sub_circuit)
    pub fn go_into(self, sub_circuit: String, lsq: usize,) -> Self {
        match self {
            CircuitPc {
                sub_circuit: None,
                pc,
                lsq: old_lsq
            } => Self::at_sub_circuit(sub_circuit, old_lsq + lsq, pc),
            circuit_pc => circuit_pc,
        }
    }

    pub fn increment(&mut self) {
        self.pc += 1
    }

    pub fn decrement(&mut self) -> bool {
        let old_pc = self.pc;
        self.pc = self.pc.saturating_sub(1);
        old_pc != self.pc
    }

    pub fn jump(&mut self, pc: usize) {
        self.pc = pc
    }

    pub fn sub_circuit(&self) -> Option<&String> {
        self.sub_circuit.as_ref()
    }

    pub fn pc(&self) -> usize {
        self.pc
    }

    pub fn lsq(&self) -> usize {
        self.lsq
    }
}
impl Display for CircuitPc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('[')?;
        if let Some(sc) = self.sub_circuit.as_ref() {
            write!(f, "{};", sc)?;
        }
        write!(f, "{}]", self.pc)
    }
}
