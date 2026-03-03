use std::ops::{Deref, Index, IndexMut};

#[derive(Debug, Clone, Default)]
pub struct BreakpointList(Vec<Breakpoint>);
impl BreakpointList {
    /// Returns PEBreakpoint::Inserted if breakpoint was inserted or PEBreakpoint::Enabled
    /// if breakpoint was enabled
    pub fn insert_or_enable(&mut self, gate_index: usize) -> IEBreakpoint {
        match self.find_breakpoint(gate_index) {
            // Breakpoint already exists
            Ok(index) => {
                self[index].enable();
                IEBreakpoint::Enabled
            }
            // Breakpoint does not exist
            Err(index) => {
                self.0.insert(index, Breakpoint::new(gate_index));
                IEBreakpoint::Inserted
            }
        }
    }

    /// Returns true if breakpoint was enabled
    pub fn enable(&mut self, gate_index: usize) -> bool {
        match self.find_breakpoint(gate_index) {
            // Breakpoint exists
            Ok(index) => {
                self[index].enable();
                true
            }
            // Breakpoint does not exist
            Err(_) => false,
        }
    }

    /// Returns true if breakpoint was disabled
    pub fn disable(&mut self, gate_index: usize) -> bool {
        match self.find_breakpoint(gate_index) {
            // Breakpoint exists
            Ok(index) => {
                self[index].disable();
                true
            }
            // Breakpoint does not exist
            Err(_) => false,
        }
    }

    /// Returns true if breakpoint was deleted
    pub fn delete(&mut self, gate_index: usize) -> bool {
        match self.find_breakpoint(gate_index) {
            // Breakpoint exists
            Ok(index) => {
                self.0.remove(index);
                true
            }
            // Breakpoint does not exist
            Err(_) => false,
        }
    }

    /// `Ok(index)` if breakpoint was found at self[index]
    ///
    /// `Err(index)` if breakpoint was not found but can be inserted
    /// at `index`
    fn find_breakpoint(&self, gate_index: usize) -> Result<usize, usize> {
        self.binary_search_by_key(&gate_index, |b| b.gate_index())
    }
}
impl Index<usize> for BreakpointList {
    type Output = Breakpoint;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for BreakpointList {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IEBreakpoint {
    Inserted,
    Enabled,
}
