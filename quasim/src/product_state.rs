use crate::{
    cart,
    ext::collapse,
    gate::{Gate, GateType, QBits},
    state_vector::StateVector,
};
use nalgebra::{Complex, dvector};
use std::ops::{Deref, DerefMut, Mul};

/// A system of potentially entangled qubits.
#[derive(Debug, Clone)]
pub struct SubSystem {
    state_vector: StateVector,
    qubits: Vec<usize>,
}
impl Mul for SubSystem {
    type Output = Self;

    /// Concatinates qubit-lists and "tensors" state vectors.
    fn mul(self, rhs: Self) -> Self::Output {
        let mut qubits = self.qubits.clone();
        qubits.extend(rhs.qubits.clone());
        Self {
            state_vector: StateVector::from(rhs.state_vector.kronecker(&self.state_vector)),
            qubits: qubits,
        }
    }
}

impl SubSystem {
    /// The system, I, that satisfies: I * x == x * I == x
    fn identity() -> Self {
        Self {
            state_vector: StateVector::from(dvector![cart!(1.0)]),
            qubits: vec![],
        }
    }

    /// The system |0>.
    pub fn zero(qubit: usize) -> Self {
        Self {
            state_vector: StateVector::from(dvector![cart!(1.0), cart!(0.0)]),
            qubits: vec![qubit],
        }
    }

    /// The system |1>.
    pub fn one(qubit: usize) -> Self {
        Self {
            state_vector: StateVector::from(dvector![cart!(0.0), cart!(1.0)]),
            qubits: vec![qubit],
        }
    }

    pub fn qubits(&self) -> &Vec<usize> {
        &self.qubits
    }

    pub fn qubits_mut(&mut self) -> &mut Vec<usize> {
        &mut self.qubits
    }

    pub fn state_vector(&self) -> &StateVector {
        &self.state_vector
    }

    pub fn state_vector_mut(&mut self) -> &mut StateVector {
        &mut self.state_vector
    }

    pub fn n_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Index of specified qubit
    pub fn local_index(&self, global_index: usize) -> usize {
        let Some(local_index) = self.qubits.iter().position(|&q| q == global_index) else {
            panic!("Qubit is not member of system!")
        };
        local_index
    }

    /// Insersion sort by qubit index and swaps on state vector accordingly.
    fn sort(&mut self) {
        let n_qubits = self.qubits.len();
        let mut i = 1;
        while i < n_qubits {
            let mut j = i;
            while j > 0 && self.qubits[j - 1] > self.qubits[j] {
                // Sort the list of qubit indecies.
                self.qubits.swap(j, j - 1);

                // Sort the state vector.
                self.state_vector
                    .apply_gate(&Gate::new(GateType::SWAP, &[], &[j, j - 1]).unwrap());

                j -= 1;
            }
            i += 1;
        }
    }
}

/// A collection of subsystems.
#[derive(Debug, Clone)]
pub struct ProductState(Vec<SubSystem>);

impl ProductState {
    /// A system with no qubits.
    pub fn empty() -> Self {
        ProductState(vec![])
    }

    /// The system |b> where b is any bitstring.
    pub fn from_bitstring(bitstring: usize, n_qubits: usize) -> Self {
        // No entanglement -> one system for each qubit.
        let mut prod = ProductState::empty();

        for i in 0..n_qubits {
            if (bitstring >> i) & 1 == 0 {
                prod.push(SubSystem::zero(i));
            } else {
                prod.push(SubSystem::one(i));
            };
        }
        prod
    }

    /// The system |0> * |0> * |0> * ...
    pub fn zeros(n_qubits: usize) -> Self {
        Self::from_bitstring(0, n_qubits)
    }

    pub fn n_qubits(&self) -> usize {
        self.iter().map(|sub| sub.n_qubits()).sum()
    }

    pub fn qubits(&self) -> Vec<usize> {
        self.clone()
            .into_iter()
            .fold(vec![], |acc, sys| vec![acc, sys.qubits].concat())
    }

    fn product(&self) -> SubSystem {
        self.clone()
            .into_iter()
            .fold(SubSystem::identity(), |acc, sys| acc * sys)
    }

    /// Returns the index for the system that contains `qubit`.
    pub fn system_of_qubit(&self, qubit: usize) -> usize {
        let Some(sys_idx) = self.iter().position(|s| s.qubits.contains(&qubit)) else {
            panic!("Qubit is not member of any system!")
        };
        sys_idx
    }

    /// The state vector of the whole system.
    /// Only works for pure states
    pub fn vector(&self) -> StateVector {
        let mut tot_sys = self.product();

        tot_sys.sort();
        tot_sys.state_vector
    }

    /// Return the amplitude for a given basis state.
    pub fn basis_value(&self, basis: usize) -> Complex<f64> {
        //Translate to order of system
        let indices = QBits::from_bitstring(basis)
            .get_indices()
            .iter()
            .map(|&inp| {
                self.qubits()
                    .iter()
                    .position(|&b| b == inp)
                    .expect("Basis should be a subset of all qubits")
            })
            .collect::<Vec<usize>>();

        let basis_translation = QBits::from_indices(&indices).get_bitstring();

        //Find the amplitude of the whole system.
        let mut amp_acc = cart!(1.0);
        let mut base_index = 0;

        for sys in self.iter() {
            let n = sys.n_qubits();
            let mask = (1 << n) - 1; // 1111..
            let local_basis = (basis_translation >> base_index) & mask;
            base_index += n;
            amp_acc *= sys.state_vector[local_basis];
        }

        amp_acc
    }

    pub fn collapse(&self) -> usize {
        let indicies = self
            .iter()
            .map(|sys| {
                QBits::from_bitstring(collapse(sys.state_vector().as_slice()))
                    .get_indices()
                    .iter()
                    .map(|&b| sys.qubits()[b])
                    .collect::<Vec<usize>>()
            })
            .collect::<Vec<Vec<usize>>>()
            .concat();

        QBits::from_indices(&indicies).get_bitstring()
    }
}

impl Deref for ProductState {
    type Target = Vec<SubSystem>;
    fn deref(&self) -> &Vec<SubSystem> {
        &self.0
    }
}
impl DerefMut for ProductState {
    fn deref_mut(&mut self) -> &mut Vec<SubSystem> {
        &mut self.0
    }
}

impl From<Vec<SubSystem>> for ProductState {
    fn from(v: Vec<SubSystem>) -> Self {
        ProductState(v)
    }
}
impl From<ProductState> for Vec<SubSystem> {
    fn from(m: ProductState) -> Vec<SubSystem> {
        m.0
    }
}

impl AsRef<Vec<SubSystem>> for ProductState {
    fn as_ref(&self) -> &Vec<SubSystem> {
        &self.0
    }
}
impl AsMut<Vec<SubSystem>> for ProductState {
    fn as_mut(&mut self) -> &mut Vec<SubSystem> {
        &mut self.0
    }
}

impl IntoIterator for ProductState {
    type Item = SubSystem;
    type IntoIter = std::vec::IntoIter<SubSystem>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl<'a> IntoIterator for &'a ProductState {
    type Item = &'a SubSystem;
    type IntoIter = std::slice::Iter<'a, SubSystem>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<'a> IntoIterator for &'a mut ProductState {
    type Item = &'a mut SubSystem;
    type IntoIter = std::slice::IterMut<'a, SubSystem>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl FromIterator<SubSystem> for ProductState {
    fn from_iter<I: IntoIterator<Item = SubSystem>>(iter: I) -> Self {
        ProductState(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        circuit::Circuit,
        ext::equal_state_c,
        product_state_simulator::ProductStateSimulator,
        product_state::ProductState,
        simulator::{BuildSimulator, DebuggableSimulator, StoredCircuitSimulator},
    };
    use nalgebra::{Complex, DVector};

    fn dvec_from_amps(state: &ProductState, n_qubits: usize) -> DVector<Complex<f64>> {
        let mut v = DVector::<Complex<f64>>::zeros(1 << n_qubits);
        for basis in 0..(1 << n_qubits) {
            v[basis] = state.basis_value(basis);
        }
        v
    }

    #[test]
    fn interleaved_amp_test() {
        let mut sim = ProductStateSimulator::build(
            Circuit::new(4)
                .h(0)
                .ch(&[0], 2)
                .swap(0, 2)
                .h(1)
                .ch(&[1], 3)
                .swap(0, 3)
                .ch(&[2], 1)
                .swap(2, 3)
                .ch(&[0], 3)
                .swap(0, 1),
        )
        .unwrap();
        while sim.next() {}

        let state = sim.state();
        let amps = dvec_from_amps(&state, sim.n_qubits());
        let dvec = state.vector();
        assert!(equal_state_c(&amps, &dvec, 4, 0.001));
    }
}
