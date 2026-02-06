use std::{
    fmt::Display,
    ops::{Add, AddAssign, Deref, Mul, MulAssign},
};

use nalgebra::{Complex, Const, DVector, Dim, Dyn, SMatrix, SVector};

type State = SVector<Complex<f32>, 2>;
type Gate = SMatrix<Complex<f32>, 2, 2>;

#[derive(Clone)]
// Matrix product of ST:s
struct MST<QS: QSystem> {
    factors: Vec<ST<QS>>,
    system: QS,
}
impl<QS: QSystem> MulAssign<MST<QS>> for MST<QS> {
    fn mul_assign(&mut self, rhs: MST<QS>) {
        self.system.assert_bit_count(rhs.system);
        self.factors.extend(rhs.factors);
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
impl<QS: QSystem> Mul<TPV<QS>> for MST<QS> {
    type Output = STV<QS>;

    fn mul(self, rhs: TPV<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);

        let mut ret: Self::Output = rhs.into();
        for st in self.iter().rev() {
            ret = st * &ret;
        }
        ret
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
struct ST<QS: QSystem> {
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
impl<QS: QSystem> Mul<TP<QS>> for ST<QS> {
    type Output = MST<QS>;

    fn mul(self, rhs: TP<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        MST {
            system: self.system,
            factors: vec![self, rhs.into()],
        }
    }
}
impl<QS: QSystem> AddAssign<ST<QS>> for ST<QS> {
    fn add_assign(&mut self, rhs: ST<QS>) {
        self.system.assert_bit_count(rhs.system);
        self.terms.extend(rhs.terms);
    }
}
impl<QS: QSystem> AddAssign<TP<QS>> for ST<QS> {
    fn add_assign(&mut self, rhs: TP<QS>) {
        self.system.assert_bit_count(rhs.system);
        self.terms.push(rhs.into())
    }
}
impl<QS: QSystem> Mul<&TPV<QS>> for &ST<QS> {
    type Output = STV<QS>;

    fn mul(self, rhs: &TPV<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        STV {
            system: self.system,
            terms: self.terms.iter().map(|tp| tp * rhs).collect(),
        }
    }
}
impl<QS: QSystem> Mul<&STV<QS>> for &ST<QS> {
    type Output = STV<QS>;

    fn mul(self, rhs: &STV<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        let mut inner = Vec::new();
        for tp in self.iter() {
            for tpv in rhs.iter() {
                inner.push(tp * tpv);
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
struct TP<QS: QSystem> {
    gates: QS::Storage<Gate>,
    system: QS,
}
impl<QS: QSystem> TP<QS> {
    pub fn new(system: QS, f: &impl Fn(usize) -> Gate) -> Self {
        TP {
            gates: system.init_gate_storage(f),
            system,
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
impl<QS: QSystem> Mul<TP<QS>> for TP<QS> {
    type Output = MST<QS>;

    fn mul(self, rhs: TP<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        MST {
            system: self.system,
            factors: vec![self.into(), rhs.into()],
        }
    }
}
impl<QS: QSystem> Mul<ST<QS>> for TP<QS> {
    type Output = MST<QS>;

    fn mul(self, rhs: ST<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        MST {
            system: self.system,
            factors: vec![self.into(), rhs],
        }
    }
}
impl<QS: QSystem> Mul<&TPV<QS>> for &TP<QS> {
    type Output = TPV<QS>;

    fn mul(self, rhs: &TPV<QS>) -> Self::Output {
        self.system.assert_bit_count(rhs.system);
        TPV {
            system: self.system,
            states: self.system.init_state_storage(&|i| self[i] * rhs[i]),
        }
    }
}
impl From<&[Gate]> for TP<Dyn> {
    fn from(value: &[Gate]) -> Self {
        Self {
            system: Dyn(value.len()),
            gates: Vec::from(value),
        }
    }
}
impl<QS: QSystem> Deref for TP<QS> {
    type Target = [Gate];

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
struct STV<QS: QSystem> {
    terms: Vec<TPV<QS>>,
    system: QS,
}
impl<QS: QSystem> STV<QS> {
    pub fn eval(&self) -> QS::SystemState {
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
struct TPV<QS: QSystem> {
    states: QS::Storage<State>,
    system: QS,
}
impl<QS: QSystem> TPV<QS> {
    pub fn new(system: QS, f: &impl Fn(usize) -> State) -> Self {
        Self {
            states: system.init_state_storage(f),
            system,
        }
    }

    pub fn eval(&self) -> QS::SystemState {
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
impl From<&[State]> for TPV<Dyn> {
    fn from(value: &[State]) -> Self {
        Self {
            system: Dyn(value.len()),
            states: Vec::from(value),
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

pub trait QSystem: Copy {
    type SystemState: Add<Self::SystemState> + AddAssign<Self::SystemState>;
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
    fn init_gate_storage(&self, f: &impl Fn(usize) -> Gate) -> Self::Storage<Gate>;
    /// All arguments passed to `f` must be in the range `0..(state_count)`
    fn init_system_state(&self, f: &impl Fn(usize) -> Complex<f32>) -> Self::SystemState;
    fn system_get(&self, system_state: &Self::SystemState, index: usize) -> Complex<f32>;
    fn system_apply_f(
        &self,
        system_state: &mut Self::SystemState,
        f: &impl Fn(usize, Complex<f32>) -> Complex<f32>,
    );

    fn bit_count(&self) -> usize;
    fn state_count(&self) -> usize {
        1 << self.bit_count()
    }
}
impl QSystem for Dyn {
    type SystemState = DVector<Complex<f32>>;
    type Storage<T: Clone> = Vec<T>;

    fn init_state_storage(&self, f: &impl Fn(usize) -> State) -> Self::Storage<State> {
        (0..self.bit_count()).into_iter().map(f).collect()
    }

    fn init_gate_storage(&self, f: &impl Fn(usize) -> Gate) -> Self::Storage<Gate> {
        (0..self.bit_count()).into_iter().map(f).collect()
    }

    fn init_system_state(&self, f: &impl Fn(usize) -> Complex<f32>) -> Self::SystemState {
        DVector::from_row_iterator(
            self.state_count(),
            (0..self.state_count()).into_iter().map(f),
        )
    }

    fn system_get(&self, system_state: &Self::SystemState, index: usize) -> Complex<f32> {
        system_state[index]
    }

    fn system_apply_f(
        &self,
        system_state: &mut Self::SystemState,
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
            type SystemState = SVector<Complex<f32>, { 1 << $qc }>;
            type Storage<T: Clone> = [T; $qc];

            fn init_state_storage(&self, f: &impl Fn(usize) -> State) -> Self::Storage<State> {
                array_init::array_init(f)
            }

            fn init_gate_storage(&self, f: &impl Fn(usize) -> Gate) -> Self::Storage<Gate> {
                array_init::array_init(f)
            }

            fn init_system_state(&self, f: &impl Fn(usize) -> Complex<f32>) -> Self::SystemState {
                SVector::from_row_iterator((0..self.state_count()).into_iter().map(f))
            }

            fn system_get(&self, system_state: &Self::SystemState, index: usize) -> Complex<f32> {
                system_state[index]
            }

            fn system_apply_f(
                &self,
                system_state: &mut Self::SystemState,
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
impl_const_qsystem!(16);
impl_const_qsystem!(17);
impl_const_qsystem!(18);
impl_const_qsystem!(19);
impl_const_qsystem!(20);
impl_const_qsystem!(21);
impl_const_qsystem!(22);
impl_const_qsystem!(23);
impl_const_qsystem!(24);
impl_const_qsystem!(25);
impl_const_qsystem!(26);
impl_const_qsystem!(27);
impl_const_qsystem!(28);
impl_const_qsystem!(29);
impl_const_qsystem!(30);
impl_const_qsystem!(31);
impl_const_qsystem!(32);
impl_const_qsystem!(33);
impl_const_qsystem!(34);
impl_const_qsystem!(35);
impl_const_qsystem!(36);
impl_const_qsystem!(37);
impl_const_qsystem!(38);
impl_const_qsystem!(39);
impl_const_qsystem!(40);
impl_const_qsystem!(41);
impl_const_qsystem!(42);
impl_const_qsystem!(43);
impl_const_qsystem!(44);
impl_const_qsystem!(45);
impl_const_qsystem!(46);
impl_const_qsystem!(47);
impl_const_qsystem!(48);
impl_const_qsystem!(49);
impl_const_qsystem!(50);
impl_const_qsystem!(51);
impl_const_qsystem!(52);
impl_const_qsystem!(53);
impl_const_qsystem!(54);
impl_const_qsystem!(55);
impl_const_qsystem!(56);
impl_const_qsystem!(57);
impl_const_qsystem!(58);
impl_const_qsystem!(59);
impl_const_qsystem!(60);
impl_const_qsystem!(61);
impl_const_qsystem!(62);
impl_const_qsystem!(63);

#[cfg(test)]
mod tests {
    use std::{
        cmp::Ordering,
        f32::{self, consts::FRAC_1_SQRT_2},
    };

    use nalgebra::{Complex, Const, DVector, Dyn, SVector};

    use crate::{
        ext::cmp_elements,
        gate_dsl::{Gate, ST, State, TP, TPV},
    };

    #[test]
    fn disp2() {
        let hadamard = Gate::new(
            FRAC_1_SQRT_2.into(),
            FRAC_1_SQRT_2.into(),
            FRAC_1_SQRT_2.into(),
            (-FRAC_1_SQRT_2).into(),
        );
        let cnot_control_0 = Gate::new(1.0.into(), 0.0.into(), 0.0.into(), 0.0.into());
        let cnot_control_1 = Gate::new(0.0.into(), 0.0.into(), 0.0.into(), 1.0.into());
        let x = Gate::new(0.0.into(), 1.0.into(), 1.0.into(), 0.0.into());

        let tp0 = TP::from([hadamard, Gate::identity(), Gate::identity()].as_slice());
        let tp1 = TP::from([cnot_control_0, Gate::identity(), Gate::identity()].as_slice());
        let tp2 = TP::from([cnot_control_1, Gate::identity(), x].as_slice());

        let st0: ST<Dyn> = tp0.into();
        let st1 = tp1 + tp2;

        let mst = st1 * st0;

        let state = TPV::from(
            [
                State::new(1.0.into(), 0.0.into()),
                State::new(1.0.into(), 0.0.into()),
                State::new(1.0.into(), 0.0.into()),
            ]
            .as_slice(),
        );
        assert_eq!(
            cmp_elements(
                &(mst * state).eval(),
                &DVector::from_row_slice(&[
                    Complex::new(FRAC_1_SQRT_2, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(FRAC_1_SQRT_2, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                ]),
                0.000001,
            ),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn disp2_const() {
        let hadamard = Gate::new(
            FRAC_1_SQRT_2.into(),
            FRAC_1_SQRT_2.into(),
            FRAC_1_SQRT_2.into(),
            (-FRAC_1_SQRT_2).into(),
        );
        let cnot_control_0 = Gate::new(1.0.into(), 0.0.into(), 0.0.into(), 0.0.into());
        let cnot_control_1 = Gate::new(0.0.into(), 0.0.into(), 0.0.into(), 1.0.into());
        let x = Gate::new(0.0.into(), 1.0.into(), 1.0.into(), 0.0.into());

        let tp0 = TP::new(Const::<3>, &|i| {
            [hadamard, Gate::identity(), Gate::identity()][i]
        });
        let tp1 = TP::new(Const::<3>, &|i| {
            [cnot_control_0, Gate::identity(), Gate::identity()][i]
        });
        let tp2 = TP::new(Const::<3>, &|i| [cnot_control_1, Gate::identity(), x][i]);

        let st0: ST<Const<3>> = tp0.into();
        let st1 = tp1 + tp2;

        let mst = st1 * st0;

        let state = TPV::new(Const::<3>, &|i| {
            [
                State::new(1.0.into(), 0.0.into()),
                State::new(1.0.into(), 0.0.into()),
                State::new(1.0.into(), 0.0.into()),
            ][i]
        });

        assert_eq!(
            cmp_elements(
                &(mst * state).eval(),
                &SVector::<Complex<f32>, 8>::from_row_slice(&[
                    Complex::new(FRAC_1_SQRT_2, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(FRAC_1_SQRT_2, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                ]),
                0.000001,
            ),
            Some(Ordering::Equal)
        );
    }
}
