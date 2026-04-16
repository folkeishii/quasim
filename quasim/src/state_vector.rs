use crate::{
    cart,
    ext::get_u_matrix2,
    gate::{Gate, GateType, QBits},
    simulator::QuantumState,
};
use nalgebra::{Complex, DMatrix, DVector, Matrix2};
use rand::distr::{Distribution, weighted::WeightedIndex};
use std::ops::{Deref, DerefMut, Index, IndexMut};

#[derive(Debug, Clone)]
pub struct StateVector(DVector<Complex<f32>>);

impl QuantumState for StateVector {
    type BasisValue = Complex<f32>;
    fn basis_value(&self, basis: usize) -> Complex<f32> {
        self[basis]
    }

    fn collapse(&self) -> usize {
        let probs = self.iter().map(|&c| c.norm_sqr());

        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }
}

impl StateVector {
    pub fn zeros(n_qubits: usize) -> Self {
        Self::from_bitstring(0, n_qubits)
    }

    /// The system |b> where b is any bitstring.
    pub fn from_bitstring(bitstring: usize, n_qubits: usize) -> Self {
        let mut v = Self(DVector::<Complex<f32>>::zeros(1 << n_qubits));
        v[bitstring] = cart!(1.0);
        v
    }

    pub fn apply_matrix(&mut self, matrix: &DMatrix<Complex<f32>>) {
        self.0 = matrix * self.0.clone();
    }

    /// Checks that all control bits are 1
    fn controls_active(i: usize, controls: QBits) -> bool {
        let control_mask = controls.get_bitstring();
        (i & control_mask) == control_mask
    }

    /// Checks that all target bits are 0
    fn is_block_base(i: usize, targets: QBits) -> bool {
        let target_mask = targets.get_bitstring();
        (i & target_mask) == 0
    }

    // 0 1
    // 1 0
    #[inline(always)]
    fn apply_x(&mut self, base_index: usize, target: QBits) {
        self.0
            .as_mut_slice()
            .swap(base_index, base_index | target.get_bitstring());
    }

    // 0 -i
    // i  0
    #[inline(always)]
    fn apply_y(&mut self, base_index: usize, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let state = self.0.as_mut_slice();
        let a = state[base_index];
        let b = state[flipped_index];

        state[base_index] = cart!(b.im, -b.re);
        state[flipped_index] = cart!(-a.im, a.re);
    }

    // 1  0
    // 0 -1
    #[inline(always)]
    fn apply_z(&mut self, base_index: usize, target: QBits) {
        let i = base_index | target.get_bitstring();
        let amp = &mut self.0[i];

        amp.re = -amp.re;
        amp.im = -amp.im;
    }

    #[inline(always)]
    fn apply_h(&mut self, base_index: usize, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let state = self.0.as_mut_slice();
        let a = state[base_index];
        let b = state[flipped_index];
        let inv_sqrt2 = 1.0 / std::f32::consts::SQRT_2;

        state[base_index] = (a + b) * inv_sqrt2;
        state[flipped_index] = (a - b) * inv_sqrt2;
    }

    #[inline(always)]
    fn apply_s(&mut self, base_index: usize, target: QBits) {
        let i = base_index | target.get_bitstring();
        let amp = self.0[i];

        self.0[i].re = -amp.im;
        self.0[i].im = amp.re;
    }

    #[inline(always)]
    fn apply_swap(&mut self, base_index: usize, targets: QBits) {
        let t0 = targets.get_indices()[0];
        let t1 = targets.get_indices()[1];

        let i01 = base_index | (1 << t0);
        let i10 = base_index | (1 << t1);

        self.0.as_mut_slice().swap(i01, i10);
    }

    #[inline(always)]
    fn apply_unitary2(&mut self, base_index: usize, u: &Matrix2<Complex<f32>>, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let a = self.0[base_index];
        let b = self.0[flipped_index];

        self.0[base_index] = u[(0, 0)] * a + u[(0, 1)] * b;
        self.0[flipped_index] = u[(1, 0)] * a + u[(1, 1)] * b;
    }

    pub fn apply_gate(&mut self, gate: &Gate) {
        let controls = gate.get_control_bits();
        let targets = gate.get_target_bits();
        let n = self.0.len();

        // No parallelization
        // State vector is length 2^n , n=num qubits
        for i in 0..n {
            if !Self::is_block_base(i, targets) {
                continue;
            }

            if !Self::controls_active(i, controls) {
                continue;
            }

            match gate.get_type() {
                GateType::X => Self::apply_x(self, i, targets),
                GateType::Y => Self::apply_y(self, i, targets),
                GateType::Z => Self::apply_z(self, i, targets),
                GateType::H => Self::apply_h(self, i, targets),
                GateType::S => Self::apply_s(self, i, targets),
                GateType::SWAP => Self::apply_swap(self, i, targets),
                GateType::U(theta, phi, lambda) => {
                    Self::apply_unitary2(self, i, &get_u_matrix2(theta, phi, lambda), targets)
                }
            }
        }
    }

    /// Measures a single qubit and updates the state vector.
    /// Returns the measured result, 0 or 1.
    pub fn measure_bit(&mut self, target: usize) -> usize {
        let mask = 1 << target;
        let measurement = self.collapse() & mask;
        let measured_bit = (measurement >> target) & 1;

        let mut norm = Complex::ZERO;

        for (i, amp) in self.0.iter_mut().enumerate() {
            if (i & mask) != measurement {
                // Remove amplitude for all states that do not align with measurement
                *amp = Complex::ZERO;
            } else {
                // If amplitude != 0, then sum
                norm += amp.norm_sqr();
            }
        }

        norm = norm.sqrt();

        // Renormalize state vector
        self.0.iter_mut().for_each(|x| *x /= norm);

        measured_bit
    }

    /// Measures all qubits and updates the state vector.
    /// Returns the measured result as a bitstring.
    pub fn measure_all(&mut self) -> usize {
        let measurement = self.collapse();

        // Collapse whole state vector
        self.0.fill(cart!(0.0));
        self.0[measurement] = cart!(1.0);

        measurement
    }

    /// # coefficent_matrix
    /// Finds the appropriate coefficent matrix
    /// used in schmidt decomposition.
    fn coefficent_matrix(&self, targets: &[usize], n_qubits: usize) -> DMatrix<Complex<f32>> {
        let squash_by_mask = |bitstring: usize, mask: usize| {
            let mut res = 0;
            let mut i = 0;
            let mut j = 0;

            while (mask >> i) != 0 {
                if (mask >> i) & 1 != 0 {
                    res |= ((bitstring >> i) & 1) << j;
                    j += 1;
                }
                i += 1;
            }

            res
        };

        let n_elements = 1 << n_qubits;
        let n_targets = targets.len();

        let mut coeffs = DMatrix::<Complex<f32>>::zeros(n_elements >> n_targets, 1 << n_targets);

        let target_mask = QBits::from_indices(targets).get_bitstring();
        let not_target_mask = (n_elements - 1) ^ target_mask;

        for i in 0..n_elements {
            let col = squash_by_mask(i, target_mask);
            let row = squash_by_mask(i, not_target_mask);
            coeffs[(row, col)] = self[i];
        }

        coeffs
    }

    /// # schmidt_trace
    /// Assuming state can be factored,
    /// returns the state with targets traced out.
    pub fn schmidt_trace(&self, targets: &[usize], n_qubits: usize) -> Self {
        // Find the coefficent matrix for the state vector.
        let coeffs = self.coefficent_matrix(targets, n_qubits);

        // Use Singular Value Decomposition to find the traced out vector.
        let svd = coeffs.svd(true, false);

        let res = DVector::<Complex<f32>>::from(svd.u.unwrap().column(0));
        // SVD might mess with global phase.
        Self(res.scale(res[0].re.signum()))
    }

    /// # schmidt_reduce
    /// Assuming state can be factored,
    /// returns the reduced state of targets.
    pub fn schmidt_reduce(&self, targets: &[usize], n_qubits: usize) -> Self {
        // Find the coefficent matrix for the state vector.
        let coeffs = self.coefficent_matrix(targets, n_qubits);

        // Use Singular Value Decomposition to find the traced out vector.
        let svd = coeffs.svd(false, true);

        let res = DVector::<Complex<f32>>::from(svd.v_t.unwrap().transpose().column(0));
        // SVD might mess with global phase.
        Self(res.scale(res[0].re.signum()))
    }
}

impl Index<usize> for StateVector {
    type Output = Complex<f32>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for StateVector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl Deref for StateVector {
    type Target = DVector<Complex<f32>>;
    fn deref(&self) -> &DVector<Complex<f32>> {
        &self.0
    }
}
impl DerefMut for StateVector {
    fn deref_mut(&mut self) -> &mut DVector<Complex<f32>> {
        &mut self.0
    }
}

impl From<DVector<Complex<f32>>> for StateVector {
    fn from(v: DVector<Complex<f32>>) -> Self {
        StateVector(v)
    }
}
impl From<StateVector> for DVector<Complex<f32>> {
    fn from(m: StateVector) -> DVector<Complex<f32>> {
        m.0
    }
}

impl AsRef<DVector<Complex<f32>>> for StateVector {
    fn as_ref(&self) -> &DVector<Complex<f32>> {
        &self.0
    }
}
impl AsMut<DVector<Complex<f32>>> for StateVector {
    fn as_mut(&mut self) -> &mut DVector<Complex<f32>> {
        &mut self.0
    }
}
impl std::fmt::Display for StateVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cart,
        ext::{equal_state_c, get_gate_matrix},
        gate::{Gate, GateType},
        state_vector::StateVector,
    };
    use nalgebra::dvector;
    use std::f32::consts::{FRAC_1_SQRT_2, PI};

    #[test]
    fn schmidt_test() {
        let hcnot = StateVector::from(dvector![
            cart!(FRAC_1_SQRT_2),
            cart!(0.0),
            cart!(0.0),
            cart!(FRAC_1_SQRT_2)
        ]);

        let rand0 = rand::random_range(-2.0 * PI..2.0 * PI);
        let rand1 = rand::random_range(-2.0 * PI..2.0 * PI);
        let rand2 = rand::random_range(-2.0 * PI..2.0 * PI);
        let zero = dvector![cart!(1.0), cart!(0.0)];
        let random_state =
            get_gate_matrix(&Gate::new(GateType::U(rand0, rand1, rand2), &[], &[0]).unwrap())
                * zero.clone();

        let tot_last = StateVector::from(random_state.kronecker(&hcnot));
        let trace_tot_last = tot_last.schmidt_trace(&[2], 3);
        assert!(equal_state_c(&trace_tot_last, &hcnot, 2, 0.001));
        let reduce_tot_last = tot_last.schmidt_reduce(&[0, 1], 3);
        assert!(equal_state_c(&reduce_tot_last, &hcnot, 2, 0.001));

        let tot_first = StateVector::from(hcnot.kronecker(&random_state));
        let trace_tot_first = tot_first.schmidt_trace(&[0], 3);
        assert!(equal_state_c(&trace_tot_first, &hcnot, 2, 0.001));
        let reduce_tot_first = tot_first.schmidt_reduce(&[1, 2], 3);
        assert!(equal_state_c(&reduce_tot_first, &hcnot, 2, 0.001));

        let mut tot_middle = StateVector::from(zero.kronecker(&random_state.kronecker(&zero)));
        tot_middle.apply_gate(&Gate::new(GateType::H, &[], &[0]).unwrap());
        tot_middle.apply_gate(&Gate::new(GateType::X, &[0], &[2]).unwrap());
        let trace_tot_middle = tot_middle.schmidt_trace(&[1], 3);
        assert!(equal_state_c(&trace_tot_middle, &hcnot, 2, 0.001));
        let reduce_tot_middle = tot_middle.schmidt_reduce(&[0, 2], 3);
        assert!(equal_state_c(&reduce_tot_middle, &hcnot, 2, 0.001));
    }
}
