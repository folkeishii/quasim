use std::ops::{Deref, Index, IndexMut};

#[derive(Debug, Clone, Default)]
pub struct BreakpointList(Vec<Breakpoint>);
impl BreakpointList {
    /// Returns true if breakpoint was inserted
    pub fn insert_or_enable(&mut self, gate_index: usize) -> PEBreakpoint {
        match self.binary_search_by_key(&gate_index, |b| b.gate_index()) {
            // Breakpoint already exists
            Ok(index) => {
                self[index].enable();
                PEBreakpoint::Enabled
            }
            // Breakpoint does not exist
            Err(index) => {
                self.inner_mut().insert(index, Breakpoint::new(gate_index));
                PEBreakpoint::Inserted
            }
        }
    }

    /// Returns true if breakpoint was enabled
    pub fn enable(&mut self, gate_index: usize) -> Option<PEBreakpoint> {
        match self.binary_search_by_key(&gate_index, |b| b.gate_index()) {
            // Breakpoint exists
            Ok(index) => {
                self[index].enable();
                Some(PEBreakpoint::Enabled)
            }
            // Breakpoint does not exist
            Err(_) => None,
        }
    }

    fn inner(&self) -> &Vec<Breakpoint> {
        &self.0
    }

    fn inner_mut(&mut self) -> &mut Vec<Breakpoint> {
        &mut self.0
    }
}
impl Index<usize> for BreakpointList {
    type Output = Breakpoint;

    fn index(&self, index: usize) -> &Self::Output {
        &self.inner()[index]
    }
}
impl IndexMut<usize> for BreakpointList {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner_mut()[index]
    }
}
impl Deref for BreakpointList {
    type Target = [Breakpoint];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Breakpoint {
    gate_index: usize,
    enabled: bool,
}
impl Breakpoint {
    pub fn new(gate_index: usize) -> Self {
        Self {
            gate_index,
            enabled: true,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn gate_index(&self) -> usize {
        self.gate_index
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }
}

pub enum PEBreakpoint {
    Inserted,
    Enabled,
}
