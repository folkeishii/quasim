use std::fmt::Display;

#[derive(Debug, Clone, Default, Hash)]
pub struct CircuitPc {
    pc: usize,
    lsq: usize,
    sub: Option<(String, Box<CircuitPc>)>,
}
impl CircuitPc {
    pub fn new(pc: usize) -> Self {
        CircuitPc {
            pc,
            lsq: 0,
            sub: None,
        }
    }

    pub fn with_lsq(pc: usize, lsq: usize) -> Self {
        CircuitPc {
            pc,
            lsq,
            sub: None,
        }
    }

    pub fn increment(&mut self) {
        *self.pc_mut() += 1
    }

    pub fn decrement(&mut self) -> bool {
        let old_pc = self.pc();
        *self.pc_mut() = self.pc.saturating_sub(1);
        old_pc != self.pc()
    }

    pub fn jump(&mut self, pc: usize) {
        self.pc = pc
    }

    pub fn jump_and_link(&mut self, name: String, lsq: usize) {
        if let Some((_, sub_pc)) = &mut self.sub {
            sub_pc.jump_and_link(name, lsq);
        } else {
            self.sub = Some((name, Box::from(CircuitPc::with_lsq(0, lsq))));
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
            let is_leaf = !sub_pc.ret();
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
