use std::ops::{Deref, Index, IndexMut};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BreakpointList(Vec<Breakpoint>);
impl BreakpointList {
    /// Returns PEBreakpoint::Inserted if breakpoint was inserted or PEBreakpoint::Enabled
    /// if breakpoint was enabled
    pub fn insert_or_enable(&mut self, pc: usize) -> IEBreakpoint {
        match self.find_breakpoint(pc) {
            // Breakpoint already exists
            Ok(index) => {
                self[index].enable();
                IEBreakpoint::Enabled
            }
            // Breakpoint does not exist
            Err(index) => {
                self.0.insert(index, Breakpoint::new(pc));
                IEBreakpoint::Inserted
            }
        }
    }

    /// Returns true if breakpoint was enabled
    pub fn enable(&mut self, pc: usize) -> bool {
        match self.find_breakpoint(pc) {
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
    pub fn disable(&mut self, pc: usize) -> bool {
        match self.find_breakpoint(pc) {
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
    pub fn delete(&mut self, pc: usize) -> bool {
        match self.find_breakpoint(pc) {
            // Breakpoint exists
            Ok(index) => {
                self.0.remove(index);
                true
            }
            // Breakpoint does not exist
            Err(_) => false,
        }
    }

    /// Returns the first breakpoint after `pc`
    pub fn next_break(&self, pc: usize) -> Option<&Breakpoint> {
        match self.find_breakpoint(pc) {
            // Breakpoint exists
            Ok(index) => self.get(index+1),
            // Breakpoint does not exist
            Err(index) => self.get(index)
        }
    }

    /// `Ok(index)` if breakpoint was found at self[index]
    ///
    /// `Err(index)` if breakpoint was not found but can be inserted
    /// at `index`
    fn find_breakpoint(&self, gate_index: usize) -> Result<usize, usize> {
        self.binary_search_by_key(&gate_index, |b| b.pc())
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

#[derive(Debug, Clone, Copy, Hash)]
pub struct Breakpoint {
    pc: usize,
    enabled: bool,
}
impl Breakpoint {
    pub fn new(pc: usize) -> Self {
        Self {
            pc,
            enabled: true,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn pc(&self) -> usize {
        self.pc
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }
}
impl PartialEq for Breakpoint {
    fn eq(&self, other: &Self) -> bool {
        self.pc == other.pc
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IEBreakpoint {
    Inserted,
    Enabled,
}
