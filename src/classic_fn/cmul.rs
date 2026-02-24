use std::ops::Mul;

use crate::{
    classic_fn::{CExpr, FnClassic},
    register::RegisterFileRef,
};

#[derive(Debug, Clone)]
pub struct CMul<T, U>(T, U);
impl<T, U, V, W, X> FnClassic for CMul<T, U>
where
    T: FnClassic<Output = V>,
    U: FnClassic<Output = W>,
    V: Mul<W, Output = X>,
{
    type Output = X;
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> X {
        self.0.classic_fn(file) * self.1.classic_fn(file)
    }
}
impl<T, U, V, W, X> Mul<CExpr<U>> for CExpr<T>
where
    T: FnClassic<Output = V>,
    U: FnClassic<Output = W>,
    V: Mul<W, Output = X>,
{
    type Output = CExpr<CMul<T, U>>;

    fn mul(self, rhs: CExpr<U>) -> Self::Output {
        CExpr::from(CMul(self.inner(), rhs.inner()))
    }
}
