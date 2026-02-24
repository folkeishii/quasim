use std::{marker::PhantomData, ops::Add};

use crate::{
    classic_fn::{CExpr2, FnClassic},
    register::RegisterFileRef,
};

#[derive(Debug, Clone)]
pub struct CAdd<T, U, V, W>(T, U, PhantomData<(V, W)>);
impl<T, U, V, W, X> FnClassic for CAdd<T, U, V, W>
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

#[derive(Debug, Clone)]
pub struct CAdd2<T, U>(T, U);
impl<T, U, V, W, X> FnClassic for CAdd2<T, U>
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
impl<T, U, V, W, X> Add<CExpr2<U>> for CExpr2<T>
where
    T: FnClassic<Output = V>,
    U: FnClassic<Output = W>,
    V: Add<W, Output = X>,
{
    type Output = CExpr2<CAdd2<T, U>>;

    fn add(self, rhs: CExpr2<U>) -> Self::Output {
        CExpr2::from(CAdd2(self.inner(), rhs.inner()))
    }
}
