mod cadd;
mod cassign;
mod cconst;
mod cexpr;
mod cle;
mod cmul;
mod cregister;
pub use cadd::*;
pub use cassign::*;
pub use cconst::*;
pub use cexpr::*;
pub use cle::*;
pub use cmul::*;
pub use cregister::*;

use std::{
    marker::PhantomData,
    ops::{Add, Mul},
};

use crate::{ext::call_unary, impl_deref, impl_deref_mut, impl_from, register::RegisterFileRef};

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
