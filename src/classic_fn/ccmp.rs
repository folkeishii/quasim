use crate::{classic_fn::FnClassic, register::RegisterFileRef};

#[derive(Debug, Clone)]
pub enum CCmp<T, U> {
    Eq(T, U),
    Ne(T, U),
    Lt(T, U),
    Le(T, U),
    Gt(T, U),
    Ge(T, U),
}
impl<T, U, V, W> FnClassic for CCmp<T, U>
where
    T: FnClassic<Output = V>,
    U: FnClassic<Output = W>,
    V: Ord + PartialOrd<W>,
{
    type Output = bool;
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> bool {
        match self {
            CCmp::Eq(lhs, rhs) => lhs.classic_fn(file).eq(&rhs.classic_fn(file)),
            CCmp::Ne(lhs, rhs) => lhs.classic_fn(file).ne(&rhs.classic_fn(file)),
            CCmp::Lt(lhs, rhs) => lhs.classic_fn(file).lt(&rhs.classic_fn(file)),
            CCmp::Le(lhs, rhs) => lhs.classic_fn(file).le(&rhs.classic_fn(file)),
            CCmp::Gt(lhs, rhs) => lhs.classic_fn(file).gt(&rhs.classic_fn(file)),
            CCmp::Ge(lhs, rhs) => lhs.classic_fn(file).ge(&rhs.classic_fn(file)),
        }
    }
}
