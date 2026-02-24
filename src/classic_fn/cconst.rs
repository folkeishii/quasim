use crate::{classic_fn::FnClassic, register::RegisterFileRef};

#[macro_export]
macro_rules! cconst {
    ($value:expr) => {
        $crate::classic_fn::CExpr2::<$crate::classic_fn::CConstant>::from(
            $crate::classic_fn::CConstant::new($value as usize),
        )
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
