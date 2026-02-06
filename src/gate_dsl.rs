use std::{
    fmt::Display,
    marker::PhantomData,
    ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, Mul, MulAssign},
};

use nalgebra::{Complex, Const, DVector, Dim, Dyn, SMatrix, SVector};

type State = SVector<Complex<f32>, 2>;
type Matrix2 = SMatrix<Complex<f32>, 2, 2>;

// Matrix product of ST:s
struct MST<const QCOUNT: usize>(Vec<ST<QCOUNT>>);
impl<const QCOUNT: usize> MST<QCOUNT> {
    pub fn new() -> MST<QCOUNT> {
        MST(vec![])
    }
}
impl<const QCOUNT: usize> MulAssign<MST<QCOUNT>> for MST<QCOUNT> {
    fn mul_assign(&mut self, rhs: MST<QCOUNT>) {
        self.0.extend(rhs.0);
    }
}
impl<const QCOUNT: usize> From<ST<QCOUNT>> for MST<QCOUNT> {
    fn from(value: ST<QCOUNT>) -> Self {
        let mut vec = Vec::with_capacity(1);
        vec.push(value);
        MST(vec)
    }
}
impl<const QCOUNT: usize> From<TP<QCOUNT>> for MST<QCOUNT> {
    fn from(value: TP<QCOUNT>) -> Self {
        let mut vec = Vec::with_capacity(1);
        vec.push(value.into());
        MST(vec)
    }
}
impl<const QCOUNT: usize> Mul<TPV<QCOUNT>> for MST<QCOUNT> {
    type Output = STV<QCOUNT>;

    fn mul(self, rhs: TPV<QCOUNT>) -> Self::Output {
        let mut ret: Self::Output = rhs.into();
        for st in self.0.iter().rev() {
            ret = st * &ret;
        }
        ret
    }
}
impl<const QCOUNT: usize> Display for MST<QCOUNT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for st in self.0.iter() {
            write!(f, "({})", st)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
// Sum of TP:s
struct ST<const QCOUNT: usize>(Vec<TP<QCOUNT>>);
impl<const QCOUNT: usize> ST<QCOUNT> {
    pub fn new<const N: usize>() -> Self {
        ST(vec![])
    }
}
impl<const QCOUNT: usize> Mul<ST<QCOUNT>> for ST<QCOUNT> {
    type Output = MST<QCOUNT>;

    fn mul(self, rhs: ST<QCOUNT>) -> Self::Output {
        MST(vec![self, rhs])
    }
}
impl<const QCOUNT: usize> Mul<TP<QCOUNT>> for ST<QCOUNT> {
    type Output = MST<QCOUNT>;

    fn mul(self, rhs: TP<QCOUNT>) -> Self::Output {
        MST(vec![self, rhs.into()])
    }
}
impl<const QCOUNT: usize> AddAssign<ST<QCOUNT>> for ST<QCOUNT> {
    fn add_assign(&mut self, rhs: ST<QCOUNT>) {
        self.0.extend(rhs.0);
    }
}
impl<const QCOUNT: usize> AddAssign<TP<QCOUNT>> for ST<QCOUNT> {
    fn add_assign(&mut self, rhs: TP<QCOUNT>) {
        self.0.push(rhs.into());
    }
}
impl<const QCOUNT: usize> Mul<TPV<QCOUNT>> for ST<QCOUNT> {
    type Output = STV<QCOUNT>;

    fn mul(self, rhs: TPV<QCOUNT>) -> Self::Output {
        STV(self.0.iter().map(|tp| tp * &rhs).collect())
    }
}
impl<const QCOUNT: usize> Mul<&TPV<QCOUNT>> for &ST<QCOUNT> {
    type Output = STV<QCOUNT>;

    fn mul(self, rhs: &TPV<QCOUNT>) -> Self::Output {
        STV(self.0.iter().map(|tp| tp * &rhs).collect())
    }
}
impl<const QCOUNT: usize> Mul<STV<QCOUNT>> for ST<QCOUNT> {
    type Output = STV<QCOUNT>;

    fn mul(self, rhs: STV<QCOUNT>) -> Self::Output {
        let mut inner = Vec::new();
        for tp in self.0.iter() {
            for tpv in rhs.0.iter() {
                inner.push(tp * tpv);
            }
        }

        STV(inner)
    }
}
impl<const QCOUNT: usize> Mul<&STV<QCOUNT>> for &ST<QCOUNT> {
    type Output = STV<QCOUNT>;

    fn mul(self, rhs: &STV<QCOUNT>) -> Self::Output {
        let mut inner = Vec::new();
        for tp in self.0.iter() {
            for tpv in rhs.0.iter() {
                inner.push(tp * tpv);
            }
        }

        STV(inner)
    }
}
impl<const QCOUNT: usize> From<TP<QCOUNT>> for ST<QCOUNT> {
    fn from(value: TP<QCOUNT>) -> Self {
        let mut vec = Vec::with_capacity(1);
        vec.push(value);
        ST(vec)
    }
}
impl<const QCOUNT: usize> Display for ST<QCOUNT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0[0])?;
        for tp in self.0.iter().skip(1) {
            write!(f, " + {}", tp)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
// Tensor Product
struct TP<const QCOUNT: usize>([Matrix2; QCOUNT]);
impl<const QCOUNT: usize> TP<QCOUNT> {
    pub fn new() -> Self {
        TP([Matrix2::identity(); QCOUNT])
    }
}
impl<const QCOUNT: usize> Add<TP<QCOUNT>> for TP<QCOUNT> {
    type Output = ST<QCOUNT>;

    fn add(self, rhs: TP<QCOUNT>) -> Self::Output {
        ST(vec![self, rhs])
    }
}
impl<const QCOUNT: usize> Mul<TP<QCOUNT>> for TP<QCOUNT> {
    type Output = MST<QCOUNT>;

    fn mul(self, rhs: TP<QCOUNT>) -> Self::Output {
        MST(vec![self.into(), rhs.into()])
    }
}
impl<const QCOUNT: usize> Mul<ST<QCOUNT>> for TP<QCOUNT> {
    type Output = MST<QCOUNT>;

    fn mul(self, rhs: ST<QCOUNT>) -> Self::Output {
        MST(vec![self.into(), rhs])
    }
}
impl<const QCOUNT: usize> Mul<TPV<QCOUNT>> for TP<QCOUNT> {
    type Output = TPV<QCOUNT>;

    fn mul(self, rhs: TPV<QCOUNT>) -> Self::Output {
        TPV(array_init::array_init(|i| self[i] * rhs.0[i]))
    }
}
impl<const QCOUNT: usize> Mul<&TPV<QCOUNT>> for &TP<QCOUNT> {
    type Output = TPV<QCOUNT>;

    fn mul(self, rhs: &TPV<QCOUNT>) -> Self::Output {
        TPV(array_init::array_init(|i| self[i] * rhs.0[i]))
    }
}
impl<const QCOUNT: usize> Deref for TP<QCOUNT> {
    type Target = [Matrix2; QCOUNT];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<const QCOUNT: usize> Display for TP<QCOUNT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self[0])?;
        for m in self.as_ref().iter().skip(1) {
            write!(f, " ⊗ {}", m)?;
        }
        Ok(())
    }
}

// Tensor Product of Vectors
struct STV<const QCOUNT: usize>(Vec<TPV<QCOUNT>>);
impl<const QCOUNT: usize> STV<QCOUNT> {
    pub fn eval(&self) -> DVector<Complex<f32>> {
        let mut ret = DVector::<Complex<f32>>::zeros(1 << QCOUNT);

        for tpv in self.0.iter() {
            ret += tpv.eval();
        }

        ret
    }
}
impl<const QCOUNT: usize> From<TPV<QCOUNT>> for STV<QCOUNT> {
    fn from(value: TPV<QCOUNT>) -> Self {
        Self(vec![value])
    }
}
impl<const QCOUNT: usize> Display for STV<QCOUNT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0[0])?;
        for tpv in self.0.iter().skip(1) {
            write!(f, " + {}", tpv)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
// Tensor Product of Vectors
struct TPV<const QCOUNT: usize>([State; QCOUNT]);
impl<const QCOUNT: usize> TPV<QCOUNT> {
    const TOTAL: usize = 1 << QCOUNT;
    pub fn eval(&self) -> DVector<Complex<f32>> {
        let mut ret = DVector::<Complex<f32>>::zeros(Self::TOTAL);
        for i in 0..Self::TOTAL {
            ret[i] = Complex { re: 1.0, im: 0.0 }
        }

        let mut stride = Self::TOTAL >> 1;
        for state in self.0 {
            let mut i = 0;
            let mut start = 0;
            while start < (1 << QCOUNT) {
                for ret_i in start..(start + stride) {
                    ret[ret_i] *= state[i & 1]
                }
                i += 1;
                start += stride;
            }
            stride >>= 1;
        }

        ret
    }
}
impl<const QCOUNT: usize> Display for TPV<QCOUNT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0[0])?;
        for v in self.0.as_ref().iter().skip(1) {
            write!(f, " ⊗ {}", v)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cmp::Ordering,
        f32::{self, consts::FRAC_1_SQRT_2},
    };

    use nalgebra::{Complex, DVector, SVector};

    use crate::{
        ext::cmp_elements,
        gate_dsl::{Matrix2, ST, State, TP, TPV},
    };

    #[test]
    fn tpv_eval() {
        let tpv = TPV([
            SVector::<Complex<f32>, 2>::new(1.0.into(), 0.0.into()),
            SVector::<Complex<f32>, 2>::new(2.0.into(), 3.0.into()),
        ]);

        assert!(
            tpv.eval()
                == SVector::<Complex<f32>, 4>::new(2.0.into(), 3.0.into(), 0.0.into(), 0.0.into())
        );

        let tpv = TPV([
            SVector::<Complex<f32>, 2>::new(2.0.into(), 3.0.into()),
            SVector::<Complex<f32>, 2>::new(1.0.into(), 0.0.into()),
        ]);

        assert!(
            tpv.eval()
                == SVector::<Complex<f32>, 4>::new(2.0.into(), 0.0.into(), 3.0.into(), 0.0.into())
        );
    }

    #[test]
    fn disp() {
        let hadamard = Matrix2::new(
            FRAC_1_SQRT_2.into(),
            FRAC_1_SQRT_2.into(),
            FRAC_1_SQRT_2.into(),
            (-FRAC_1_SQRT_2).into(),
        );
        let cnot_control_0 = Matrix2::new(1.0.into(), 0.0.into(), 0.0.into(), 0.0.into());
        let cnot_control_1 = Matrix2::new(0.0.into(), 0.0.into(), 0.0.into(), 1.0.into());
        let x = Matrix2::new(0.0.into(), 1.0.into(), 1.0.into(), 0.0.into());

        let tp0 = TP([hadamard, Matrix2::identity(), Matrix2::identity()]);
        let tp1 = TP([cnot_control_0, Matrix2::identity(), Matrix2::identity()]);
        let tp2 = TP([cnot_control_1, Matrix2::identity(), x]);

        let st0: ST<3> = tp0.into();
        let st1 = tp1 + tp2;

        let mst = st1 * st0;

        let state = TPV([
            State::new(1.0.into(), 0.0.into()),
            State::new(1.0.into(), 0.0.into()),
            State::new(1.0.into(), 0.0.into()),
        ]);
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
}
