use crate::ext::{OrdByKey, SortedVec};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BreakpointList(SortedVec<Breakpoint, usize>);
impl BreakpointList {
    /// Returns PEBreakpoint::Inserted if breakpoint was inserted or PEBreakpoint::Enabled
    /// if breakpoint was enabled
    pub fn insert_or_enable(&mut self, pc: usize) -> IEBreakpoint {
        match self.0.insert(Breakpoint::new(pc)) {
            Some(_) => IEBreakpoint::Enabled,
            None => IEBreakpoint::Inserted,
        }
    }

    /// Returns true if breakpoint was enabled
    pub fn enable(&mut self, pc: usize) -> bool {
        self.0.map(&pc, &mut |br| br.enable())
    }

    /// Returns true if breakpoint was disabled
    pub fn disable(&mut self, pc: usize) -> bool {
        self.0.map(&pc, &mut |br| br.disable())
    }

    /// Returns true if breakpoint was deleted
    pub fn delete(&mut self, pc: usize) -> bool {
        self.0.remove(&pc).is_some()
    }

    /// Returns the first breakpoint after `pc`
    pub fn next_break(&self, pc: usize) -> Option<&Breakpoint> {
        self.0.get_or_next(&(pc + 1))
    }

    pub fn get(&self, pc: usize) -> Option<&Breakpoint> {
        self.0.get(&pc)
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct Breakpoint {
    pc: usize,
    enabled: bool,
}
impl Breakpoint {
    pub fn new(pc: usize) -> Self {
        Self { pc, enabled: true }
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
impl OrdByKey<usize> for Breakpoint {
    fn key(&self) -> &usize {
        &self.pc
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IEBreakpoint {
    Inserted,
    Enabled,
}
