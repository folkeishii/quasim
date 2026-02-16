use std::{
    fmt::Display,
    ops::{Add, AddAssign, Deref, Mul, MulAssign},
};

use nalgebra::{
    ArrayStorage, Complex, Const, DMatrix, DVector, Dim, Dyn, Matrix, SMatrix, SVector, StorageMut,
    VecStorage,
};

use crate::{cart, ext::reverse_indices};

pub type State = SVector<Complex<f32>, 2>;
pub type Gate2x2 = SMatrix<Complex<f32>, 2, 2>;

pub const STATE_0: State = SMatrix::from_array_storage(ArrayStorage([[cart!(1.0), cart!(0.0)]]));
pub const ID: Gate2x2 = SMatrix::from_array_storage(ArrayStorage([
    [cart!(1.0), cart!(0.0)],
    [cart!(0.0), cart!(1.0)],
]));
pub const AT_00: Gate2x2 = SMatrix::from_array_storage(ArrayStorage([
    [cart!(1.0), cart!(0.0)],
    [cart!(0.0), cart!(0.0)],
]));
pub const AT_01: Gate2x2 = SMatrix::from_array_storage(ArrayStorage([
    [cart!(0.0), cart!(0.0)],
    [cart!(1.0), cart!(0.0)],
]));
pub const AT_10: Gate2x2 = SMatrix::from_array_storage(ArrayStorage([
    [cart!(0.0), cart!(1.0)],
    [cart!(0.0), cart!(0.0)],
]));
pub const AT_11: Gate2x2 = SMatrix::from_array_storage(ArrayStorage([
    [cart!(0.0), cart!(0.0)],
    [cart!(0.0), cart!(1.0)],
]));

#[derive(Clone)]
// Matrix product of ST:s
pub struct MST<QS: QSystem> {
    factors: Vec<ST<QS>>,
    system: QS,
}
impl<QS: QSystem> MST<QS> {
    pub fn eval(
        &self,
    ) -> Matrix<Complex<f32>, <QS as QSystem>::SysR, <QS as QSystem>::SysC, <QS as QSystem>::SysGS>
    {
        let system = self.system;
        let mut stm: Vec<TP<QS>> = vec![TP {
            system,
            gates: system.init_gate_storage(&|_| ID),
        }];

        for i in reverse_indices(0..self.len(), self.len()) {
            let st = &self[i];
            let prev = stm.clone();
            for _ in 1..st.len() {
                stm.extend(prev.clone());
            }

            let stride = prev.len();
            for j in 0..stm.len() {
                // let mul_lhs = &st[j / stride];
                // let mul_rhs = &stm[j];

                for k in 0..system.bit_count() {
                    stm[j].gates.as_mut()[k] = st[j / stride][k] * stm[j].as_ref()[k];
                }
            }
        }

        let mut ret = system.init_system_gate(&|_, _| cart!(0.0));
        for tm in stm {
            ret += tm.eval();
        }

        ret
    }
}
impl<QS: QSystem> Mul<MST<QS>> for MST<QS> {
    type Output = MST<QS>;

    fn mul(self, rhs: MST<QS>) -> Self::Output {
        let mut out = self;
        out *= rhs;
        out
    }
}
impl<QS: QSystem> MulAssign<MST<QS>> for MST<QS> {
    fn mul_assign(&mut self, rhs: MST<QS>) {
        self.system.assert_bit_count(rhs.system);
        self.factors.extend(rhs.factors);
    }
}
impl<QS: QSystem> Mul<TPV<QS>> for MST<QS> {
    type Output = STV<QS>;

    fn mul(self, rhs: TPV<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);

        let mut ret: Self::Output = rhs.into();
        for st in self.iter().rev() {
            ret = st.clone() * ret;
        }
        ret
    }
}
impl<QS: QSystem> From<ST<QS>> for MST<QS> {
    fn from(value: ST<QS>) -> Self {
        MST {
            system: value.system,
            factors: vec![value],
        }
    }
}
impl<QS: QSystem> From<TP<QS>> for MST<QS> {
    fn from(value: TP<QS>) -> Self {
        MST {
            system: value.system,
            factors: vec![value.into()],
        }
    }
}
impl<QS: QSystem> Deref for MST<QS> {
    type Target = [ST<QS>];

    fn deref(&self) -> &Self::Target {
        self.factors.as_slice()
    }
}
impl<QS: QSystem> Display for MST<QS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for st in self.iter() {
            write!(f, "({})", st)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
// Sum of TP:s
pub struct ST<QS: QSystem> {
    terms: Vec<TP<QS>>,
    system: QS,
}
impl<QS: QSystem> Mul<ST<QS>> for ST<QS> {
    type Output = MST<QS>;

    fn mul(self, rhs: ST<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        MST {
            system: self.system,
            factors: vec![self, rhs],
        }
    }
}
impl<QS: QSystem> Add<ST<QS>> for ST<QS> {
    type Output = Self;

    fn add(self, rhs: ST<QS>) -> Self::Output {
        let mut out = self;
        out += rhs;
        out
    }
}
impl<QS: QSystem> AddAssign<ST<QS>> for ST<QS> {
    fn add_assign(&mut self, rhs: ST<QS>) {
        self.system.assert_bit_count(rhs.system);
        self.terms.extend(rhs.terms);
    }
}
impl<QS: QSystem> Mul<TPV<QS>> for ST<QS> {
    type Output = STV<QS>;

    fn mul(self, rhs: TPV<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        STV {
            system: self.system,
            terms: self.terms.into_iter().map(|tp| tp * rhs.clone()).collect(),
        }
    }
}
impl<QS: QSystem> Mul<STV<QS>> for ST<QS> {
    type Output = STV<QS>;

    fn mul(self, rhs: STV<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        let mut inner = Vec::new();
        for tp in self.iter() {
            for tpv in rhs.iter() {
                inner.push(tp.clone() * tpv.clone());
            }
        }

        STV {
            system: self.system,
            terms: inner,
        }
    }
}
impl<QS: QSystem> From<TP<QS>> for ST<QS> {
    fn from(value: TP<QS>) -> Self {
        ST {
            system: value.system,
            terms: vec![value],
        }
    }
}
impl<QS: QSystem> Deref for ST<QS> {
    type Target = [TP<QS>];

    fn deref(&self) -> &Self::Target {
        self.terms.as_slice()
    }
}
impl<QS: QSystem> Display for ST<QS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self[0])?;
        for tp in self.iter().skip(1) {
            write!(f, " + {}", tp)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
// Tensor Product
pub struct TP<QS: QSystem> {
    gates: QS::Storage<Gate2x2>,
    system: QS,
}
impl<QS: QSystem> TP<QS> {
    pub fn eval(
        &self,
    ) -> Matrix<Complex<f32>, <QS as QSystem>::SysR, <QS as QSystem>::SysC, <QS as QSystem>::SysGS>
    {
        let stride_init = self.system.state_count() >> 1;
        self.system.init_system_gate(&|row, col| {
            let mut ret = cart!(1.0);
            let mut stride = stride_init;
            for i in 0..self.system.bit_count() {
                let row2x2 = (row / stride) & 1;
                let col2x2 = (col / stride) & 1;
                ret *= self[i][(row2x2, col2x2)];
                stride >>= 1;
            }
            ret
        })
    }
}
impl<QS: SQsystem> TP<QS> {
    pub fn from_array(array: QS::Storage<Gate2x2>) -> Self {
        Self {
            gates: array,
            system: QS::system(),
        }
    }
}
impl<QS: QSystem> Add<TP<QS>> for TP<QS> {
    type Output = ST<QS>;

    fn add(self, rhs: TP<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        ST {
            system: self.system,
            terms: vec![self, rhs],
        }
    }
}
impl<QS: QSystem> Mul<TPV<QS>> for TP<QS> {
    type Output = TPV<QS>;

    fn mul(self, rhs: TPV<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        TPV {
            system: self.system,
            states: self.system.init_state_storage(&|i| self[i] * rhs[i]),
        }
    }
}
// impl<const N: usize> From<[Gate2x2; N]> for TP<Const<N>> defined in impl_const_qsystem!
impl From<Vec<Gate2x2>> for TP<Dyn> {
    fn from(value: Vec<Gate2x2>) -> Self {
        Self {
            system: Dyn(value.len()),
            gates: value,
        }
    }
}
impl<QS: QSystem> Deref for TP<QS> {
    type Target = [Gate2x2];

    fn deref(&self) -> &Self::Target {
        self.gates.as_ref()
    }
}
impl<QS: QSystem> Display for TP<QS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self[0])?;
        for m in self.as_ref().iter().skip(1) {
            write!(f, " ⊗ {}", m)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
// Sum of TPV2s
pub struct STV<QS: QSystem> {
    terms: Vec<TPV<QS>>,
    system: QS,
}
impl<QS: QSystem> STV<QS> {
    pub fn eval(
        &self,
    ) -> Matrix<Complex<f32>, <QS as QSystem>::SysR, Const<1>, <QS as QSystem>::SysSS> {
        let mut ret = self.system.init_system_state(&|_| 0.0.into());

        for tpv in self.iter() {
            ret += tpv.eval();
        }

        ret
    }
}
impl<QS: QSystem> From<TPV<QS>> for STV<QS> {
    fn from(value: TPV<QS>) -> Self {
        Self {
            system: value.system,
            terms: vec![value],
        }
    }
}
impl<QS: QSystem> Deref for STV<QS> {
    type Target = [TPV<QS>];

    fn deref(&self) -> &Self::Target {
        self.terms.as_slice()
    }
}
impl<QS: QSystem> Display for STV<QS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self[0])?;
        for tpv in self.iter().skip(1) {
            write!(f, " + {}", tpv)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
// Tensor Product of Vectors
pub struct TPV<QS: QSystem> {
    states: QS::Storage<State>,
    system: QS,
}
impl<QS: QSystem> TPV<QS> {
    pub fn eval(
        &self,
    ) -> Matrix<Complex<f32>, <QS as QSystem>::SysR, Const<1>, <QS as QSystem>::SysSS> {
        let mut ret = self
            .system
            .init_system_state(&|_| Complex { re: 1.0, im: 0.0 });

        let mut stride = self.system.state_count() >> 1;
        for state in self.iter() {
            self.system
                .system_apply_f(&mut ret, &|i, el| el * state[(i / stride) & 1]);
            stride >>= 1;
        }

        ret
    }
}
impl<QS: SQsystem> TPV<QS> {
    pub fn from_array(array: QS::Storage<State>) -> Self {
        Self {
            states: array,
            system: QS::system(),
        }
    }
}
// impl<const N: usize> From<[State; N]> for TPV<Const<N>> defined in impl_const_qsystem!
impl From<Vec<State>> for TPV<Dyn> {
    fn from(value: Vec<State>) -> Self {
        Self {
            system: Dyn(value.len()),
            states: value,
        }
    }
}
impl<QS: QSystem> Deref for TPV<QS> {
    type Target = [State];

    fn deref(&self) -> &Self::Target {
        self.states.as_ref()
    }
}
impl<QS: QSystem> Display for TPV<QS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self[0])?;
        for v in self.as_ref().iter().skip(1) {
            write!(f, " ⊗ {}", v)?;
        }
        Ok(())
    }
}

pub trait StateTrait: Sized + Add<Self> + AddAssign<Self> {}
impl<T: Sized + Add<Self> + AddAssign<Self>> StateTrait for T {}
pub trait GateTrait<State>:
    Sized + Add<Self> + AddAssign<Self> + Mul<Self> + MulAssign<Self> + Mul<State>
{
}
impl<T: Sized + Add<Self> + AddAssign<Self> + Mul<Self> + MulAssign<Self> + Mul<State>, State>
    GateTrait<State> for T
{
}

pub trait QSystem: Copy {
    type SysR: Dim;
    type SysC: Dim;
    type SysSS: StorageMut<Complex<f32>, Self::SysR, Const<1>>;
    type SysGS: StorageMut<Complex<f32>, Self::SysR, Self::SysC>;
    type Storage<T: Clone>: Clone + AsRef<[T]> + AsMut<[T]>;

    fn assert_bit_count(self, other: Self) {
        assert_eq!(self.bit_count(), other.bit_count());
    }

    /// All arguments passed to `f` must be in the range `0..(bit_count)`
    ///
    /// Returned storage must be of length `bit_count()`
    fn init_state_storage(&self, f: &impl Fn(usize) -> State) -> Self::Storage<State>;
    /// All arguments passed to `f` must be in the range `0..(bit_count)`
    ///
    /// Returned storage must be of length `bit_count()`
    fn init_gate_storage(&self, f: &impl Fn(usize) -> Gate2x2) -> Self::Storage<Gate2x2>;
    /// All arguments passed to `f` must be in the range `0..(state_count)`
    fn init_system_state(
        &self,
        f: &impl Fn(usize) -> Complex<f32>,
    ) -> Matrix<Complex<f32>, Self::SysR, Const<1>, Self::SysSS>;
    /// All arguments passed to `f` must be in the range `0..(state_count) x 0..(state_count)`
    fn init_system_gate(
        &self,
        f: &impl Fn(usize, usize) -> Complex<f32>,
    ) -> Matrix<Complex<f32>, Self::SysR, Self::SysC, Self::SysGS>;
    fn system_get(
        &self,
        system_state: &Matrix<Complex<f32>, Self::SysR, Const<1>, Self::SysSS>,
        index: usize,
    ) -> Complex<f32>;
    fn system_apply_f(
        &self,
        system_state: &mut Matrix<Complex<f32>, Self::SysR, Const<1>, Self::SysSS>,
        f: &impl Fn(usize, Complex<f32>) -> Complex<f32>,
    );

    fn bit_count(&self) -> usize;
    fn state_count(&self) -> usize {
        1 << self.bit_count()
    }
}

pub trait SQsystem: QSystem {
    fn system() -> Self;
}

impl QSystem for Dyn {
    type SysR = Dyn;
    type SysC = Dyn;
    type SysSS = VecStorage<Complex<f32>, Self::SysR, Const<1>>;
    type SysGS = VecStorage<Complex<f32>, Self::SysR, Self::SysC>;
    type Storage<T: Clone> = Vec<T>;

    fn init_state_storage(&self, f: &impl Fn(usize) -> State) -> Self::Storage<State> {
        (0..self.bit_count()).into_iter().map(f).collect()
    }

    fn init_gate_storage(&self, f: &impl Fn(usize) -> Gate2x2) -> Self::Storage<Gate2x2> {
        (0..self.bit_count()).into_iter().map(f).collect()
    }

    fn init_system_state(&self, f: &impl Fn(usize) -> Complex<f32>) -> DVector<Complex<f32>> {
        DVector::from_row_iterator(
            self.state_count(),
            (0..self.state_count()).into_iter().map(f),
        )
    }

    fn init_system_gate(&self, f: &impl Fn(usize, usize) -> Complex<f32>) -> DMatrix<Complex<f32>> {
        let row_total = self.state_count();
        let total = self.state_count().pow(2);
        DMatrix::from_row_iterator(
            row_total,
            row_total,
            (0..total).into_iter().map(|i| {
                let (row, col) = (i / row_total, i % row_total);
                f(row, col)
            }),
        )
    }

    fn system_get(&self, system_state: &DVector<Complex<f32>>, index: usize) -> Complex<f32> {
        system_state[index]
    }

    fn system_apply_f(
        &self,
        system_state: &mut DVector<Complex<f32>>,
        f: &impl Fn(usize, Complex<f32>) -> Complex<f32>,
    ) {
        for i in 0..self.state_count() {
            system_state[i] = f(i, system_state[i])
        }
    }

    fn bit_count(&self) -> usize {
        self.value()
    }
}

macro_rules! impl_const_qsystem {
    ($qc:expr) => {
        impl QSystem for Const<$qc> {
            type SysR = Const<{ 1 << $qc }>;
            type SysC = Const<{ 1 << $qc }>;
            type SysSS = ArrayStorage<Complex<f32>, { 1 << $qc }, 1>;
            type SysGS = ArrayStorage<Complex<f32>, { 1 << $qc }, { 1 << $qc }>;
            type Storage<T: Clone> = [T; $qc];

            fn init_state_storage(&self, f: &impl Fn(usize) -> State) -> Self::Storage<State> {
                array_init::array_init(f)
            }

            fn init_gate_storage(&self, f: &impl Fn(usize) -> Gate2x2) -> Self::Storage<Gate2x2> {
                array_init::array_init(f)
            }

            fn init_system_state(
                &self,
                f: &impl Fn(usize) -> Complex<f32>,
            ) -> SVector<Complex<f32>, { 1 << $qc }> {
                SVector::from_row_iterator((0..self.state_count()).into_iter().map(f))
            }

            fn init_system_gate(
                &self,
                f: &impl Fn(usize, usize) -> Complex<f32>,
            ) -> SMatrix<Complex<f32>, { 1 << $qc }, { 1 << $qc }> {
                let row_total = self.state_count();
                let total = self.state_count().pow(2);
                SMatrix::from_row_iterator((0..total).into_iter().map(|i| {
                    let (row, col) = (i / row_total, i % row_total);
                    f(row, col)
                }))
            }

            fn system_get(
                &self,
                system_state: &SVector<Complex<f32>, { 1 << $qc }>,
                index: usize,
            ) -> Complex<f32> {
                system_state[index]
            }

            fn system_apply_f(
                &self,
                system_state: &mut SVector<Complex<f32>, { 1 << $qc }>,
                f: &impl Fn(usize, Complex<f32>) -> Complex<f32>,
            ) {
                for i in 0..self.state_count() {
                    system_state[i] = f(i, system_state[i])
                }
            }

            fn bit_count(&self) -> usize {
                $qc
            }
        }

        impl SQsystem for Const<$qc> {
            fn system() -> Self {
                Const::<$qc>
            }
        }

        impl From<[Gate2x2; $qc]> for TP<Const<$qc>> {
            fn from(value: [Gate2x2; $qc]) -> Self {
                Self {
                    system: Const::<$qc>,
                    gates: value,
                }
            }
        }

        impl From<[State; $qc]> for TPV<Const<$qc>> {
            fn from(value: [State; $qc]) -> Self {
                Self {
                    system: Const::<$qc>,
                    states: value,
                }
            }
        }
    };
}

impl_const_qsystem!(1);
impl_const_qsystem!(2);
impl_const_qsystem!(3);
impl_const_qsystem!(4);
impl_const_qsystem!(5);
impl_const_qsystem!(6);
impl_const_qsystem!(7);
impl_const_qsystem!(8);
impl_const_qsystem!(9);
impl_const_qsystem!(10);
impl_const_qsystem!(11);
impl_const_qsystem!(12);
impl_const_qsystem!(13);
impl_const_qsystem!(14);
impl_const_qsystem!(15);

// There is a chance that following
// impls do not compile depending
// on the targets available memory
//
// impl_const_qsystem!(16);
// impl_const_qsystem!(17);
// impl_const_qsystem!(18);
// impl_const_qsystem!(19);
// impl_const_qsystem!(20);
// impl_const_qsystem!(21);
// impl_const_qsystem!(22);
// impl_const_qsystem!(23);
// impl_const_qsystem!(24);
// impl_const_qsystem!(25);
// impl_const_qsystem!(26);
// impl_const_qsystem!(27);
// impl_const_qsystem!(28);
// impl_const_qsystem!(29);
// impl_const_qsystem!(30);
// impl_const_qsystem!(31);
// impl_const_qsystem!(32);

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_1_SQRT_2;

    use nalgebra::{Complex, Const, DVector, Dyn, SVector};

    use crate::{
        cart,
        ext::equal_to_matrix_c,
        gate_dsl::{AT_00, AT_11, Gate2x2, ID, ST, STATE_0, TP, TPV},
    };

    #[test]
    fn hadamard_cnot_entanglement() {
        let hadamard = Gate2x2::new(
            cart!(FRAC_1_SQRT_2),
            cart!(FRAC_1_SQRT_2),
            cart!(FRAC_1_SQRT_2),
            cart!(-FRAC_1_SQRT_2),
        );
        let x = Gate2x2::new(cart!(0.0), cart!(1.0), cart!(1.0), cart!(0.0));

        let tp0 = TP::from(vec![hadamard, ID, ID]);
        let tp1 = TP::from(vec![AT_00, ID, ID]);
        let tp2 = TP::from(vec![AT_11, ID, x]);

        let st0: ST<Dyn> = tp0.into();
        let st1 = tp1 + tp2;

        let mst = st1 * st0;

        let state = TPV::from(vec![STATE_0, STATE_0, STATE_0]);
        assert!(equal_to_matrix_c(
            &(mst * state).eval(),
            &DVector::from_row_slice(&[
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
            ]),
            0.000001,
        ));
    }

    #[test]
    fn hadamard_cnot_entanglement_stack() {
        let hadamard = Gate2x2::new(
            FRAC_1_SQRT_2.into(),
            FRAC_1_SQRT_2.into(),
            FRAC_1_SQRT_2.into(),
            (-FRAC_1_SQRT_2).into(),
        );
        let x = Gate2x2::new(0.0.into(), 1.0.into(), 1.0.into(), 0.0.into());

        let tp0 = TP::from([hadamard, ID, ID]);
        let tp1 = TP::from([AT_00, ID, ID]);
        let tp2 = TP::from([AT_11, ID, x]);

        let st0: ST<Const<3>> = tp0.into();
        let st1 = tp1 + tp2;

        let mst = st1 * st0;

        let state = TPV::from([STATE_0, STATE_0, STATE_0]);

        assert!(equal_to_matrix_c(
            &(mst * state).eval(),
            &SVector::from_row_slice(&[
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
            ]),
            0.000001,
        ));
    }

    #[test]
    fn hadamard_cnot_entanglement_final_matrix() {
        let hadamard = Gate2x2::new(
            FRAC_1_SQRT_2.into(),
            FRAC_1_SQRT_2.into(),
            FRAC_1_SQRT_2.into(),
            (-FRAC_1_SQRT_2).into(),
        );
        let x = Gate2x2::new(0.0.into(), 1.0.into(), 1.0.into(), 0.0.into());

        let tp0 = TP::from([hadamard, ID, ID]);
        let tp1 = TP::from([AT_00, ID, ID]);
        let tp2 = TP::from([AT_11, ID, x]);

        let st0: ST<Const<3>> = tp0.into();
        let st1 = tp1 + tp2;

        let mst = st1 * st0;

        let final_matrix = mst.eval();
        let state = TPV::from([STATE_0, STATE_0, STATE_0]).eval();

        assert!(equal_to_matrix_c(
            &(final_matrix * state),
            &SVector::from_row_slice(&[
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
            ]),
            0.000001,
        ));
    }
}
