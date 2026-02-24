mod cadd;
mod cassign;
mod ccmp;
mod cconst;
mod cexpr;
mod cmul;
mod cregister;
pub use cadd::*;
pub use cassign::*;
pub use ccmp::*;
pub use cconst::*;
pub use cexpr::*;
pub use cmul::*;
pub use cregister::*;

use crate::{ext::call_unary, register::RegisterFileRef};

pub trait FnClassic {
    type Output;
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> Self::Output;
}
impl<F: Clone + Fn(&mut RegisterFileRef<usize>) -> T, T> FnClassic for F {
    type Output = T;

    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> T {
        call_unary(self, file)
    }
}
