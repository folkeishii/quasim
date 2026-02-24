use std::marker::PhantomData;

use crate::{classic_fn::FnClassic, register::RegisterFileRef};

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

// pub trait ClassicLe<U> {
//     type Output;
//     fn le(self, rhs: U) -> Self::Output;
// }
// impl<T: PartialOrd<U>, U> ClassicLe<U> for T {
//     type Output = bool;

//     fn le(self, rhs: U) -> Self::Output {
//         self <= rhs
//     }
// }

// #[derive(Debug, Clone)]
// pub struct CLe2<T, U>(T, U);
// impl<T, U, V, W> FnClassic for CLe2<T, U>
// where
//     T: FnClassic<Output = V>,
//     U: FnClassic<Output = W>,
//     V: ClassicLe<W>,
// {
//     type Output = bool;
//     fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> bool {
//         self.0.classic_fn(file).le(self.1.classic_fn(file))
//     }
// }
// impl<T, U, V, W, X> Mul<CExpr2<U>> for CExpr2<T>
// where
//     T: FnClassic<Output = V>,
//     U: FnClassic<Output = W>,
//     V: Mul<W, Output = X>,
// {
//     type Output = CExpr2<CMul2<T, U>>;

//     fn mul(self, rhs: CExpr2<U>) -> Self::Output {
//         CExpr2::from(CMul2(self.inner(), rhs.inner()))
//     }
// }
