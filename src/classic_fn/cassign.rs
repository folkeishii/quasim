use crate::{
    classic_fn::{CRegister, FnClassic},
    register::RegisterFileRef,
};

#[derive(Debug, Clone)]
pub struct CAssign<T>(pub(crate) CRegister, pub(crate) T);
impl<T: FnClassic<Output = usize>> FnClassic for CAssign<T> {
    type Output = ();
    fn classic_fn(&self, file: &mut RegisterFileRef<usize>) -> () {
        file[&self.0] = self.1.classic_fn(file)
    }
}
