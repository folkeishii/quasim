use std::ops::Add;

use crate::{
    classic_fn::{CExpr, FnClassic},
    register::RegisterFileRef,
};

#[derive(Debug, Clone)]
pub struct CAdd<T, U>(T, U);
impl<T, U, V, W, X> FnClassic for CAdd<T, U>
where
    T: FnClassic<Output = V>,
    U: FnClassic<Output = W>,
    V: Add<W, Output = X>,
{
    type Output = X;
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> X {
        self.0.classic_fn(file) + self.1.classic_fn(file)
    }
}
impl<T, U, V, W, X> Add<CExpr<U>> for CExpr<T>
where
    T: FnClassic<Output = V>,
    U: FnClassic<Output = W>,
    V: Add<W, Output = X>,
{
    type Output = CExpr<CAdd<T, U>>;

    fn add(self, rhs: CExpr<U>) -> Self::Output {
        CExpr::from(CAdd(self.inner(), rhs.inner()))
    }
}
