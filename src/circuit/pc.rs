use std::fmt::Display;

#[derive(Debug, Clone, Default, Hash)]
pub struct CircuitPc {
    pc: usize,
    lsq: usize,
}
impl CircuitPc {
    pub fn new(pc: usize) -> Self {
        CircuitPc { pc, lsq: 0 }
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

    pub fn pc(&self) -> usize {
        self.pc
    }

    pub fn lsq(&self) -> usize {
        self.lsq
    }
}
impl PartialEq for CircuitPc {
    fn eq(&self, other: &Self) -> bool {
        self.pc == other.pc
    }
}
impl Display for CircuitPc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.pc)
    }
}
