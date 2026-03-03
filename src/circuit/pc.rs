use std::fmt::{Display, Write};


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CircuitPc {
    sub_circuit: Option<String>,
    pc: usize,
}
impl CircuitPc {
    pub fn at_main(pc: usize) -> Self {
        CircuitPc {
            sub_circuit: None,
            pc,
        }
    }

    pub fn at_sub_circuit(sub_circuit: String, pc: usize) -> Self {
        CircuitPc {
            sub_circuit: Some(sub_circuit),
            pc,
        }
    }

    /// Maps a None into Some(sub_circuit)
    pub fn map_sub_circuit(self, sub_circuit: String) -> Self {
        match self {
            CircuitPc {
                sub_circuit: None,
                pc,
            } => Self::at_sub_circuit(sub_circuit, pc),
            circuit_pc => circuit_pc,
        }
    }

    pub fn sub_circuit(&self) -> Option<&String> {
        self.sub_circuit.as_ref()
    }

    pub fn pc(&self) -> usize {
        self.pc
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
