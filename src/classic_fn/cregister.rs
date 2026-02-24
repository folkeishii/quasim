use crate::{
    classic_fn::{CExpr, FnClassic},
    impl_deref,
    register::RegisterFileRef,
};

#[macro_export]
macro_rules! c {
    ($name:expr) => {
        $crate::classic_fn::CExpr::<$crate::classic_fn::CRegister>::from(
            $crate::classic_fn::CRegister::new($name.into()),
        )
    };
}

#[derive(Debug, Clone)]
pub struct CRegister(String);
impl CRegister {
    pub fn new(name: String) -> CExpr<Self> {
        Self(name).into()
    }
}
impl_deref!(CRegister(str));
impl FnClassic for CRegister {
    type Output = usize;
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> usize {
        file[&self.0]
    }
}
