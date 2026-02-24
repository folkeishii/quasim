use std::{
    marker::PhantomData,
    ops::{Add, Mul},
};

use crate::{
    classic_fn::{CAdd, CAssign, CLE, CMul, CRegister, FnClassic},
    impl_deref, impl_deref_mut, impl_from,
    register::RegisterFileRef,
};

#[derive(Debug, Clone)]
pub struct CExpr<T, V>(T, PhantomData<V>);
impl<T: FnClassic<Output = V>, V> FnClassic for CExpr<T, V> {
    type Output = V;
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> V {
        self.0.classic_fn(file)
    }
}
impl_deref_mut!(CExpr<T, V>);
impl_from!(CExpr<T, V>, Default::default());
pub struct CExpr2<T>(T);
impl<T> CExpr2<T> {
    pub(crate) fn inner(self) -> T {
        self.0
    }
}
impl<T: FnClassic<Output = V>, V> FnClassic for CExpr2<T> {
    type Output = V;
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> V {
        self.0.classic_fn(file)
    }
}
impl_deref_mut!(CExpr2<T>);
impl_from!(CExpr2<T>);
// impl CExpr<CRegister, usize> {
//     pub fn assign<U, W>(self, rhs: CExpr<U, usize>) -> CExpr<CAssign<U>, ()>
//     where
//         U: FnClassic<Output = W>,
//     {
//         CExpr(CAssign(self.0, rhs.0), Default::default())
//     }
// }
// impl<T: FnClassic<Output = V>, V> CExpr<T, V> {
//     pub fn le<U, W>(self, rhs: CExpr<U, W>) -> CExpr<CLE<T, U, V, W>, bool>
//     where
//         U: FnClassic<Output = W>,
//         V: PartialOrd<W>,
//     {
//         CExpr(CLE(self.0, rhs.0, Default::default()), Default::default())
//     }
// }
// impl<T, U, V, W, X> Add<CExpr<U, W>> for CExpr<T, V>
// where
//     T: FnClassic<Output = V>,
//     U: FnClassic<Output = W>,
//     V: Add<W, Output = X>,
// {
//     type Output = CExpr<CAdd<T, U, V, W>, X>;

//     fn add(self, rhs: CExpr<U, W>) -> Self::Output {
//         CExpr(CAdd(self.0, rhs.0, Default::default()), Default::default())
//     }
// }
// impl<T, U, V, W, X> Mul<CExpr<U, W>> for CExpr<T, V>
// where
//     T: FnClassic<Output = V>,
//     U: FnClassic<Output = W>,
//     V: Mul<W, Output = X>,
// {
//     type Output = CExpr<CMul<T, U, V, W>, X>;

//     fn mul(self, rhs: CExpr<U, W>) -> Self::Output {
//         CExpr(CMul(self.0, rhs.0, Default::default()), Default::default())
//     }
// }
