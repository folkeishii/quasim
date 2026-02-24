use crate::{
    classic_fn::{CAssign, CCmp, CRegister, FnClassic},
    impl_deref, impl_deref_mut, impl_from,
    register::RegisterFileRef,
};
pub struct CExpr<T>(T);
impl<T> CExpr<T> {
    pub(crate) fn inner(self) -> T {
        self.0
    }
}
impl<T: FnClassic<Output = V>, V> FnClassic for CExpr<T> {
    type Output = V;
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> V {
        self.0.classic_fn(file)
    }
}
impl_deref_mut!(CExpr<T>);
impl_from!(CExpr<T>);
/// Assign ops
impl CExpr<CRegister> {
    pub fn assign<T>(self, other: CExpr<T>) -> CExpr<CAssign<T>>
    where
        T: FnClassic<Output = usize>,
    {
        CExpr::from(CAssign(self.inner(), other.inner()))
    }
}

/// Eq ops
impl<T> CExpr<T> {
    pub fn eq<U, V, W>(self, other: CExpr<U>) -> CExpr<CCmp<T, U>>
    where
        T: FnClassic<Output = V>,
        U: FnClassic<Output = W>,
        V: PartialEq<W>,
    {
        CExpr::from(CCmp::Eq(self.inner(), other.inner()))
    }

    pub fn ne<U, V, W>(self, other: CExpr<U>) -> CExpr<CCmp<T, U>>
    where
        T: FnClassic<Output = V>,
        U: FnClassic<Output = W>,
        V: PartialEq<W>,
    {
        CExpr::from(CCmp::Ne(self.inner(), other.inner()))
    }
}

/// Ord ops
impl<T> CExpr<T> {
    pub fn lt<U, V, W>(self, other: CExpr<U>) -> CExpr<CCmp<T, U>>
    where
        T: FnClassic<Output = V>,
        U: FnClassic<Output = W>,
        V: Ord + PartialOrd<W>,
    {
        CExpr::from(CCmp::Lt(self.inner(), other.inner()))
    }
    pub fn le<U, V, W>(self, other: CExpr<U>) -> CExpr<CCmp<T, U>>
    where
        T: FnClassic<Output = V>,
        U: FnClassic<Output = W>,
        V: Ord + PartialOrd<W>,
    {
        CExpr::from(CCmp::Le(self.inner(), other.inner()))
    }
    pub fn gt<U, V, W>(self, other: CExpr<U>) -> CExpr<CCmp<T, U>>
    where
        T: FnClassic<Output = V>,
        U: FnClassic<Output = W>,
        V: Ord + PartialOrd<W>,
    {
        CExpr::from(CCmp::Gt(self.inner(), other.inner()))
    }
    pub fn ge<U, V, W>(self, other: CExpr<U>) -> CExpr<CCmp<T, U>>
    where
        T: FnClassic<Output = V>,
        U: FnClassic<Output = W>,
        V: Ord + PartialOrd<W>,
    {
        CExpr::from(CCmp::Ge(self.inner(), other.inner()))
    }
}
