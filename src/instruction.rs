use std::{fmt::Debug, sync::Arc};

use crate::{
    classic_fn::FnClassic,
    gate::{Gate, QBits},
};

#[derive(Clone)]
pub enum Instruction {
    Gate(Gate),
    Measurement(QBits),
    FnClassic(Arc<dyn FnClassic<Output = ()>>),
    JumpIf(Arc<dyn FnClassic<Output = bool>>, String),
    Jump(String),
}
impl Debug for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gate(arg0) => f.debug_tuple("Gate").field(arg0).finish(),
            Self::Measurement(arg0) => f.debug_tuple("Measurement").field(arg0).finish(),
            Self::FnClassic(_) => f.debug_tuple("FnClassic").field(&"...").finish(),
            Self::JumpIf(_, arg1) => f.debug_tuple("JumpIf").field(&"...").field(arg1).finish(),
            Self::Jump(arg0) => f.debug_tuple("Jump").field(arg0).finish(),
        }
    }
}
