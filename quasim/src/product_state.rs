use crate::{cart, ext::swap_matrix, gate::QBits};
use nalgebra::{Complex, DVector, dvector};

/// A system of potentially entangled qubits.
#[derive(Debug, Clone)]
pub struct SubSystem {
    pub state: DVector<Complex<f64>>,
    pub qubits: Vec<usize>,
}

/// A collection of subsystems.
#[derive(Debug, Clone)]
pub struct ProductState {
    pub systems: Vec<SubSystem>,
}

impl SubSystem {
    /// Concatinates qubit-lists and "tensors" state vectors.
    pub fn combine(&self, rhs: &Self) -> Self {
        let mut qubits = self.qubits.clone();
        qubits.extend(rhs.qubits.clone());
        Self {
            state: rhs.state.kronecker(&self.state),
            qubits: qubits,
        }
    }

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
                self.state = swap_matrix(&[], j, j - 1, n_qubits) * self.state.clone();

                j -= 1;
            }
            i += 1;
        }
    }
}

impl std::fmt::Display for SubSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QUBITS: {:?}, STATE: {}", self.qubits, self.state)
    }
}

impl std::fmt::Display for ProductState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = self.systems.iter().map(|s| format!("{}", s)).collect();
        write!(f, "{}", parts.join(", "))
    }
}

impl Into<DVector<Complex<f64>>> for ProductState {
    fn into(self) -> DVector<Complex<f64>> {
        self.vector()
    }
}

impl ProductState {
    pub fn from_bitstring(bitstring: usize, n_qubits: usize) -> Self {
        // No entanglement -> one system for each qubit.
        let mut sys = vec![];

        for i in 0..n_qubits {
            let state = if (bitstring >> i) & 1 == 0 {
                dvector![cart!(1.0), cart!(0.0)] // |0>
            } else {
                dvector![cart!(0.0), cart!(1.0)] // |1>
            };
            sys.push(SubSystem {
                state: state,
                qubits: vec![i],
            })
        }
        Self { systems: sys }
    }

    pub fn zeros(n_qubits: usize) -> Self {
        Self::from_bitstring(0, n_qubits)
    }

    pub fn get_subsystem(&self, index: usize) -> SubSystem {
        self.systems[index].clone()
    }

    fn product(&self) -> SubSystem {
        self.systems.clone().into_iter().fold(
            SubSystem {
                state: dvector![cart!(1.0)],
                qubits: vec![],
            },
            |acc, sys| acc.combine(&sys),
        )
    }

    pub fn system_of_qubit(&self, qubit: usize) -> usize {
        let Some(sys_idx) = self.systems.iter().position(|s| s.qubits.contains(&qubit)) else {
            panic!("Qubit is not member of any system!")
        };
        sys_idx
    }

    /// The state vector of the whole system.
    /// Only works for pure states
    pub fn vector(&self) -> DVector<Complex<f64>> {
        let mut tot_sys = self.product();

        tot_sys.sort();
        tot_sys.state
    }

    pub fn amp_at(&self, basis: usize) -> Complex<f64> {
        //Translate to order of system
        let product_order = self
            .systems
            .iter()
            .fold(vec![], |acc, sys| vec![acc, sys.qubits.clone()].concat());

        let translation = QBits::from_indices(
            &QBits::from_bitstring(basis)
                .get_indices()
                .iter()
                .map(|&inp| {
                    product_order
                        .iter()
                        .position(|&b| b == inp)
                        .expect("Basis should be a subset of all qubits")
                })
                .collect::<Vec<usize>>(),
        )
        .get_bitstring();

        //Find the amplitude of the whole system.
        let mut amp_acc = cart!(1.0);
        let mut base_index = 0;

        for sys in &self.systems {
            let n = sys.qubits.len();
            let mask = (1 << n) - 1; // 1111..
            let amp_idx = (translation >> base_index) & mask;
            base_index += n;
            amp_acc *= sys.state[amp_idx];
        }

        amp_acc
    }
}

mod tests {
    use crate::{
        cart,
        circuit::Circuit,
        ext::equal_state_c,
        prod_simulator::ProdSimulator,
        product_state::ProductState,
        simulator::{BuildSimulator, DebuggableSimulator, StoredCircuitSimulator},
    };
    use nalgebra::{Complex, DVector};

    fn dvec_from_amps(state: &ProductState, n_qubits: usize) -> DVector<Complex<f64>> {
        let mut v = DVector::<Complex<f64>>::zeros(1 << n_qubits);
        for basis in 0..(1 << n_qubits) {
            v[basis] = state.amp_at(basis);
        }
        v
    }

    #[test]
    pub fn interleaved_amp_test() {
        let mut sim = ProdSimulator::build(
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

        let state = sim.get_state();
        let amps = dvec_from_amps(&state, sim.n_qubits());
        let dvec = state.vector();
        assert!(equal_state_c(&amps, &dvec, 4, 0.001));
    }
}
