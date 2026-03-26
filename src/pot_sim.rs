use std::{ops::Index};

use nalgebra::Complex;

use crate::{
    cart,
    circuit::{Circuit, HybridCircuit, pc::CircuitPc},
    gate::QBits,
};

pub struct PotSim {
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
}

enum QString {
    Branch(Box<QBranch>),
    State(QState),
}

impl QString {
    pub fn new(height: usize, index: QBits, val: Complex<f64>, zero_margin: f64) -> Self {
        if height > 1 {
            QString::Branch(Box::from(QBranch::new(height, index, val, zero_margin)))
        } else {
            QString::State(QState::new(index, val, zero_margin))
        }
    }

    pub fn set(&mut self, index: QBits, val: Complex<f64>) {
        match self {
            QString::Branch(qbranch) => qbranch.set(index, val),
            QString::State(qstate) => qstate.set(index, val),
        }
    }

    pub fn height(&self) -> usize {
        match self {
            QString::Branch(qbranch) => qbranch.height,
            QString::State(_) => 1,
        }
    }

    pub fn alive(&self) -> bool {
        match self {
            QString::Branch(qbranch) => qbranch.alive(),
            QString::State(qstate) => qstate.alive(),
        }
    }
}

impl Index<QBits> for QString {
    type Output = Complex<f64>;

    fn index(&self, index: QBits) -> &Self::Output {
        match self {
            QString::Branch(bb) => &bb[index],
            QString::State(st) => &st[index],
        }
    }
}

struct QBranch {
    height: usize,
    /// If a state is less than `zero_margin` it is
    /// considered improbable
    zero_margin: f64,
    z: Option<QString>,
    o: Option<QString>,
}

impl QBranch {
    pub fn new(height: usize, index: QBits, val: Complex<f64>, zero_margin: f64) -> Self {
        if state_zero(index) {
            QBranch {
                height,
                zero_margin,
                z: Some(QString::new(height - 1, index >> 1, val, zero_margin)),
                o: None,
            }
        } else {
            QBranch {
                height,
                zero_margin,
                z: None,
                o: Some(QString::new(height - 1, index >> 1, val, zero_margin)),
            }
        }
    }

    pub fn set(&mut self, index: QBits, val: Complex<f64>) {
        // Acquire 0 or 1
        if state_zero(index) {
            Self::set_aux(&mut self.z, self.height, self.zero_margin, index, val);
        } else {
            Self::set_aux(&mut self.o, self.height, self.zero_margin, index, val);
        };
    }

    pub fn alive(&self) -> bool {
        self.z.is_some() || self.o.is_some()
    }

    fn set_aux(
        qs_opt: &mut Option<QString>,
        height: usize,
        zero_margin: f64,
        index: QBits,
        val: Complex<f64>,
    ) {
        if let Some(qs) = qs_opt.as_mut() {
            qs.set(index >> 1, val);
            if !qs.alive() {
                *qs_opt = None
            }
        } else if val.norm() >= zero_margin {
            *qs_opt = Some(QString::new(height - 1, index >> 1, val, zero_margin))
        }
    }
}

impl Index<QBits> for QBranch {
    type Output = Complex<f64>;

    fn index(&self, index: QBits) -> &Self::Output {
        if state_zero(index) {
            self.o
                .as_ref()
                .map(|qs| &qs[index >> 1])
                .unwrap_or(&cart!(0))
        } else {
            self.o
                .as_ref()
                .map(|qs| &qs[index >> 1])
                .unwrap_or(&cart!(0))
        }
    }
}

struct QState {
    /// If a state is less than `zero_margin` it is
    /// considered improbable
    zero_margin: f64,
    z: Option<Box<Complex<f64>>>,
    o: Option<Box<Complex<f64>>>,
}

impl QState {
    pub fn new(index: QBits, val: Complex<f64>, zero_margin: f64) -> Self {
        if state_zero(index) {
            QState {
                zero_margin,
                z: Some(Box::from(val)),
                o: None,
            }
        } else {
            QState {
                zero_margin,
                z: None,
                o: Some(Box::from(val)),
            }
        }
    }

    pub fn set(&mut self, index: QBits, val: Complex<f64>) {
        // Acquire 0 or 1
        let state = if state_zero(index) {
            &mut self.z.as_mut().map(|st| **st)
        } else {
            &mut self.o.as_mut().map(|st| **st)
        };

        // Set 0 or 1 to val
        if val.norm() < self.zero_margin {
            *state = None
        } else if let Some(state) = state {
            *state = val
        } else {
            *state = Some(val)
        }
    }

    pub fn alive(&self) -> bool {
        self.z.is_some() || self.o.is_some()
    }
}

impl Index<QBits> for QState {
    type Output = Complex<f64>;

    fn index(&self, index: QBits) -> &Self::Output {
        if state_zero(index) {
            self.z.as_ref().map(|st| &**st).unwrap_or(&cart!(0))
        } else {
            self.o.as_ref().map(|st| &**st).unwrap_or(&cart!(0))
        }
    }
}

//Pair of states
//Where only one qubit differs
// pub struct StatePair

#[inline(always)]
const fn state_zero(index: QBits) -> bool {
    index.inner() & 1 == 0
}
