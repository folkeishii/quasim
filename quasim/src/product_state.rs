use crate::{
    cart,
    ext::{expand_matrix_from_gate, measure_and_observe_sv, reduced_state, swap_matrix},
    gate::{Gate, GateType},
};
use nalgebra::{Complex, DVector, dvector};

/// A system of potentially entangled qubits.
#[derive(Debug, Clone)]
struct EntSys {
    state: DVector<Complex<f64>>,
    qubits: Vec<usize>,
}

impl std::fmt::Display for EntSys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Qubits: {:?}\n Density: {}", self.qubits, self.state)
    }
}

impl EntSys {
    /// Concatinates qubit-lists and "tensors" state vectors.
    fn add_system(&self, rhs: &Self) -> Self {
        let mut qubits = self.qubits.clone();
        qubits.extend(rhs.qubits.clone());
        Self {
            state: rhs.state.kronecker(&self.state),
            qubits: qubits,
        }
    }

    fn to_local_index(&self, global_index: usize) -> usize {
        let Some(local_index) = self.qubits.iter().position(|&q| q == global_index) else {
            panic!("Qubit is not member of system!")
        };
        local_index
    }

    /// Insersion sort by qubit index and swaps on state vector :w
    /// accordingly.
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

/// A system of potentially entangled qubits.
#[derive(Debug, Clone)]
pub struct ProductState {
    systems: Vec<EntSys>,
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
            sys.push(EntSys {
                state: state,
                qubits: vec![i],
            })
        }
        Self { systems: sys }
    }

    pub fn zeros(n_qubits: usize) -> Self {
        Self::from_bitstring(0, n_qubits)
    }

    fn product(&self) -> EntSys {
        self.systems.clone().into_iter().fold(
            EntSys {
                state: dvector![cart!(1.0)],
                qubits: vec![],
            },
            |acc, sys| acc.add_system(&sys),
        )
    }

    fn find_system_of_qubit(&self, qubit: usize) -> usize {
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

    pub fn collapse_qubit(&mut self, target: usize) -> usize {
        /* Idea:
         *      - Once a qubit is measured and observed
         *      it can not be entangled with any other
         *      system.
         *      --> Measurements "splits" the measured system.
         *
         *      ex: 3 qubits A,B,C that might be entangled, A measured to |0>
         *
         *                      p` == |0><0| * Tr_BC(p)
         * */

        let target_system = self.find_system_of_qubit(target);
        let mut sys = self.systems[target_system].clone();

        // Translate global qubit indexing to the system's local indexing.
        let local_target = sys.to_local_index(target);
        let local_n_qubits = sys.qubits.len();
        let local_non_targets: Vec<usize> =
            (0..local_n_qubits).filter(|&i| i != local_target).collect();

        let (measurement, post_measure_state) =
            measure_and_observe_sv(local_target, &sys.state, local_n_qubits);

        // "Split" state.
        sys.qubits.remove(local_target);
        let post_measure_state_adj = post_measure_state.adjoint();
        let density = post_measure_state * post_measure_state_adj; //|s><s|
        sys.state = DVector::from(
            reduced_state(&density, &local_non_targets, local_n_qubits)
                .symmetric_eigen()
                .eigenvectors
                .column(0),
        );
        self.systems[target_system] = sys;

        let collapsed_state = if measurement == 0 {
            dvector![cart!(1.0), cart!(0.0)] // |0>
        } else {
            dvector![cart!(0.0), cart!(1.0)] // |1>
        };

        self.systems.push(EntSys {
            state: collapsed_state,
            qubits: vec![target],
        });

        measurement
    }

    pub fn apply_gate(&mut self, gate: Gate) {
        /* Overview:
         *
         *      - if gate acts on a single system,
         *      then apply gate to that system only.
         *
         *      - if gate acts on multiple systems,
         *      then combine those systems and then
         *      apply gate to the total system.
         *      (Exception: SWAP gates, they do not
         *      cause entanglement)
         *
         * */

        let controls = gate.get_controls();
        let targets = gate.get_targets();

        // Qubits that gate acts on.
        let mut gate_qubits = controls.clone();
        gate_qubits.extend(&targets);

        // Systems that gate acts on.
        let mut gate_systems = vec![];
        for qubit in gate_qubits {
            let sys_idx = self.find_system_of_qubit(qubit);
            if !gate_systems.contains(&sys_idx) {
                gate_systems.push(sys_idx);
            }
        }

        let mut sys = self.systems[gate_systems[0]].clone();

        let gate_acts_on_several_systems = gate_systems.len() > 1;
        let is_swap_gate = gate.get_type() == GateType::SWAP && controls.is_empty();

        if gate_acts_on_several_systems && is_swap_gate {
            // Regular swaps do not result in entanglement.
            // Only change global qubit index.
            let sys1_local_target = sys.to_local_index(targets[0]);
            let mut sys2 = self.systems[gate_systems[1]].clone();
            let sys2_local_target = sys2.to_local_index(targets[1]);

            // Swap.
            let temp = sys.qubits[sys1_local_target];
            sys.qubits[sys1_local_target] = sys2.qubits[sys2_local_target];
            sys2.qubits[sys2_local_target] = temp;

            self.systems[gate_systems[0]] = sys;
            self.systems[gate_systems[1]] = sys2;
            return;
        } else if gate_acts_on_several_systems {
            // Combine all systems acted on.
            sys = gate_systems
                .iter()
                .skip(1)
                .map(|&qs_idx| self.systems[qs_idx].clone())
                .fold(sys, |acc, qs| acc.add_system(&qs));
        }

        // Translate global qubit indexing to the system's local indexing.
        let local_targets: Vec<usize> = targets.iter().map(|&t| sys.to_local_index(t)).collect();

        if is_swap_gate {
            // Swap qubit indecies, no need for matrix multiplication.
            sys.qubits.swap(local_targets[0], local_targets[1]);
            self.systems[gate_systems[0]] = sys;
            return;
        }

        // Continue to translate global qubit indexing to the system's local indexing.
        let local_controls: Vec<usize> = controls.iter().map(|&t| sys.to_local_index(t)).collect();
        let local_n_qubits = sys.qubits.len();
        let local_gate = Gate::new(gate.get_type(), &local_controls, &local_targets).unwrap();

        // Apply the gate to the system's state vector using: |s>` == U|s>
        let mat = expand_matrix_from_gate(&local_gate, local_n_qubits);
        sys.state = mat * sys.state;

        if gate_acts_on_several_systems {
            // Remove systems that were combined.
            self.systems = self
                .systems
                .iter()
                .enumerate()
                .filter(|(i, _)| !gate_systems.contains(i))
                .map(|(_, s)| s.clone())
                .collect();

            // Add combined system.
            self.systems.push(sys);
            return;
        }
        // Update system acted on.
        self.systems[gate_systems[0]] = sys;
    }
}
