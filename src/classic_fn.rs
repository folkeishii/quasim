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
impl CExpr<CRegister, usize> {
    pub fn assign<U, W>(self, rhs: CExpr<U, usize>) -> CExpr<CAssign<U>, ()>
    where
        U: FnClassic<Output = W>,
    {
        CExpr(CAssign(self.0, rhs.0), Default::default())
    }
}
impl<T: FnClassic<Output = V>, V> CExpr<T, V> {
    pub fn le<U, W>(self, rhs: CExpr<U, W>) -> CExpr<CLE<T, U, V, W>, bool>
    where
        U: FnClassic<Output = W>,
        V: PartialOrd<W>,
    {
        CExpr(CLE(self.0, rhs.0, Default::default()), Default::default())
    }
}
impl<T, U, V, W, X> Add<CExpr<U, W>> for CExpr<T, V>
where
    T: FnClassic<Output = V>,
    U: FnClassic<Output = W>,
    V: Add<W, Output = X>,
{
    type Output = CExpr<CAdd<T, U, V, W>, X>;

    fn add(self, rhs: CExpr<U, W>) -> Self::Output {
        CExpr(CAdd(self.0, rhs.0, Default::default()), Default::default())
    }
}
impl<T, U, V, W, X> Mul<CExpr<U, W>> for CExpr<T, V>
where
    T: FnClassic<Output = V>,
    U: FnClassic<Output = W>,
    V: Mul<W, Output = X>,
{
    type Output = CExpr<CMul<T, U, V, W>, X>;

    fn mul(self, rhs: CExpr<U, W>) -> Self::Output {
        CExpr(CMul(self.0, rhs.0, Default::default()), Default::default())
    }
}

#[macro_export]
macro_rules! c {
    ($name:expr) => {
        CExpr::from(CRegister::new($name.into()))
    };
}
#[macro_export]
macro_rules! cconst {
    ($value:expr) => {
        CExpr::from(CConstant::new($value as usize))
    };
}
#[derive(Debug, Clone)]
pub struct CConstant(usize);
impl CConstant {
    pub fn new(value: usize) -> Self {
        Self(value)
    }
}
impl FnClassic for CConstant {
    type Output = usize;
    fn classic_fn(&self, _: &mut RegisterFileRef<usize>) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct CRegister(String);
impl CRegister {
    pub fn new(name: String) -> Self {
        Self(name)
    }

    pub fn assign<T: FnClassic<Output = usize>>(self, value: T) -> CAssign<T> {
        CAssign(self, value)
    }
}
impl FnClassic for CRegister {
    type Output = usize;
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> usize {
        file[&self.0]
    }
}

#[derive(Debug, Clone)]
pub struct CMul<T, U, V, W>(T, U, PhantomData<(V, W)>);
impl<T, U, V, W, X> FnClassic for CMul<T, U, V, W>
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
pub struct CLE<T, U, V, W>(T, U, PhantomData<(V, W)>);
impl<T, U, V, W> FnClassic for CLE<T, U, V, W>
where
    T: FnClassic<Output = V>,
    U: FnClassic<Output = W>,
    V: PartialOrd<W>,
{
    type Output = bool;
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> bool {
        self.0.classic_fn(file) <= self.1.classic_fn(file)
    }
}

#[derive(Debug, Clone)]
pub struct CAssign<T>(CRegister, T);
impl<T: FnClassic<Output = usize>> FnClassic for CAssign<T> {
    type Output = ();
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> () {
        file[&self.0.0] = self.1.classic_fn(file)
    }
}
