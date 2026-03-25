use std::fmt::Display;

use crate::gate::QBits;

#[derive(Debug, Clone, Default, Hash)]
pub struct CircuitPc {
    pc: usize,
    lsq: usize,
    ctrl: QBits,
    sub: Option<(String, Box<CircuitPc>)>,
}
impl CircuitPc {
    pub fn new(pc: usize) -> Self {
        CircuitPc {
            pc,
            lsq: 0,
            ctrl: 0.into(),
            sub: None,
        }
    }

    pub fn with_lsq(pc: usize, lsq: usize) -> Self {
        CircuitPc {
            pc,
            lsq,
            ctrl: 0.into(),
            sub: None,
        }
    }

    pub fn with_ctrl(pc: usize, lsq: usize, ctrl: QBits) -> Self {
        CircuitPc {
            pc,
            lsq,
            ctrl,
            sub: None,
        }
    }

    pub fn increment(&mut self) {
        *self.pc_mut() += 1
    }

    pub fn decrement(&mut self) -> bool {
        let old_pc = self.pc();
        *self.pc_mut() = old_pc.saturating_sub(1);
        old_pc != self.pc()
    }

    pub fn jump(&mut self, pc: usize) {
        *self.pc_mut() = pc
    }

    pub fn jump_and_link(&mut self, name: String, lsq: usize, ctrl: QBits) {
        if let Some((_, sub_pc)) = &mut self.sub {
            sub_pc.jump_and_link(name, lsq, ctrl);
        } else {
            self.sub = Some((
                name,
                Box::from(CircuitPc::with_ctrl(0, self.lsq + lsq, self.ctrl | ctrl)),
            ));
        }
    }

    pub fn ret(&mut self) -> bool {
        if let Some((_, sub_pc)) = &mut self.sub {
            let is_leaf = !sub_pc.ret();
            if is_leaf {
                self.sub = None;
                self.pc += 1;
            }
            true
        } else {
            false
        }
    }

    pub fn ret_backwards(&mut self) -> bool {
        if let Some((_, sub_pc)) = &mut self.sub {
            let is_leaf = !sub_pc.ret_backwards();
            if is_leaf {
                self.sub = None;
            }
            true
        } else {
            false
        }
    }

    pub fn pc(&self) -> usize {
        if let Some((_, sub_pc)) = &self.sub {
            sub_pc.pc()
        } else {
            self.pc
        }
    }

    fn pc_mut(&mut self) -> &mut usize {
        if let Some((_, sub_pc)) = &mut self.sub {
            sub_pc.pc_mut()
        } else {
            &mut self.pc
        }
    }

    pub(crate) fn next_sub_pc(&self) -> Option<(&str, &CircuitPc)> {
        if let Some((name, sub_pc)) = &self.sub {
            Some((name.as_str(), sub_pc.as_ref()))
        } else {
            None
        }
    }

    pub fn lsq(&self) -> usize {
        if let Some((_, pc)) = self.next_sub_pc() {
            pc.lsq()
        } else {
            self.lsq
        }
    }

    pub fn ctrl(&self) -> QBits {
        if let Some((_, pc)) = self.next_sub_pc() {
            pc.ctrl()
        } else {
            self.ctrl
        }
    }

    pub fn current(&self) -> (Option<&str>, usize) {
        if let Some((name, pc)) = self.sub.as_ref() {
            pc.current_aux(name)
        } else {
            (None, self.pc)
        }
    }

    fn current_aux<'a>(&'a self, with_name: &'a str) -> (Option<&'a str>, usize) {
        if let Some((name, pc)) = self.sub.as_ref() {
            pc.current_aux(name)
        } else {
            (Some(with_name), self.pc)
        }
    }

    pub fn depth(&self) -> usize {
        if let Some((_, pc)) = self.sub.as_ref() {
            pc.depth() + 1
        } else {
            1
        }
    }
}
impl PartialEq for CircuitPc {
    fn eq(&self, other: &Self) -> bool {
        self.pc == other.pc
    }
}

impl Display for CircuitPc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (sc, pc) = self.current();
        match (sc, pc) {
            (None, pc) => write!(f, "[{}]", pc),
            (Some(sc), pc) => write!(f, "[{}; {}]", sc, pc),
        }
    }
}
