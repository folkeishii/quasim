use std::{cell::OnceCell, collections::HashMap, fmt::Debug};

use log::debug;
use rand::random;

use crate::{
    gate::{Gate, GateType, QBits},
    simulator::QuantumState,
    syntax_simulator::{
        basis::{ExtendedBasis, ExtendedQubitBasis},
        scalar::Scalar,
    },
};

#[derive(Clone, PartialEq)]
pub struct ScaledState(pub ExtendedBasis, pub Scalar);

impl ScaledState {
    pub fn all_inherent_states(&self) -> Sum {
        let my_scalar = self.1;

        self.0
            .clone()
            .all_inherent_states()
            .iter()
            // Guaranteed usize means cheap clone.
            .map(|substate| my_scalar * substate.clone())
            .collect()
    }

    pub fn has_zero_coef(&self) -> bool {
        self.1.is_zero()
    }
}

impl From<(usize, Scalar)> for ScaledState {
    fn from(value: (usize, Scalar)) -> Self {
        let basis = value.0;
        let scalar = value.1;
        ScaledState(ExtendedBasis::Binary(basis), scalar)
    }
}

impl Debug for ScaledState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}{:?}", self.1, self.0)
    }
}

pub type Sum = Vec<ScaledState>;

#[derive(Debug, Clone)]
pub struct SumOfScaledStates {
    pub n_qubits: usize,
    pub sum: Sum,
    probability_distribution_cache: OnceCell<HashMap<usize, Scalar>>,
}

impl SumOfScaledStates {
    pub fn guaranteed_full_zero(n_qubits: usize) -> Self {
        let n_zeros = ScaledState(ExtendedBasis::Binary(0), Scalar::ONE);
        Self {
            n_qubits,
            sum: vec![n_zeros],
            probability_distribution_cache: OnceCell::new(),
        }
    }

    /// Returns the scaled states over all basis states
    /// that exist at this point. Extended bases (eg, |+⟩, |−⟩, |i⟩, |−i⟩)
    /// will not be expanded, so the resulting states may still contain extended bases.
    /// If you want to get the distribution over only binary states,
    /// use `all_states_unsugared` instead.
    pub fn all_scaled_states(&self) -> impl Iterator<Item = &ScaledState> {
        self.sum.iter()
    }

    pub fn apply_gate(&mut self, gate: &Gate) {
        self.probability_distribution_cache.take(); // Invalidate the cached distribution

        let controls = gate.get_control_bits();
        let targets = gate.get_target_bits().get_indices();

        debug!(
            "Applying gate {:?} with controls {:?} and target {:?}",
            gate.get_type(),
            controls,
            targets
        );

        use GateType::*;
        if matches!(gate.get_type(), GateType::SWAP) {
            let lsb = [targets[0]];
            let msb = [targets[1]];
            self.apply_gate(&Gate::new(X, &msb, &lsb).unwrap());
            self.apply_gate(&Gate::new(X, &lsb, &msb).unwrap());
            self.apply_gate(&Gate::new(X, &msb, &lsb).unwrap());
            return;
        }

        assert_eq!(targets.len(), 1);
        let only_target = targets[0];

        self.sum = match gate.get_type() {
            X => self.apply_controlled_gate(ExtendedBasis::x, controls, only_target),
            Y => self.apply_controlled_gate(ExtendedBasis::y, controls, only_target),
            Z => self.apply_controlled_gate(ExtendedBasis::z, controls, only_target),
            H => self.apply_controlled_gate(ExtendedBasis::h, controls, only_target),
            S => self.apply_controlled_gate(ExtendedBasis::s, controls, only_target),
            SWAP => {
                panic!("SWAP gates are handled separately and should not reach this point")
            }
            U(theta, phi, lambda) => self.apply_u_gate(controls, only_target, theta, phi, lambda),
        };
    }

    pub fn apply_controlled_gate(
        &self,
        unconditional_single_qubit_gate: fn(&mut ExtendedBasis, usize) -> Option<Scalar>,
        controls: QBits,
        target: usize,
    ) -> Sum {
        let mut result = self.expand_necessary_controls(controls);

        result
            .iter_mut()
            // Keep only the states where all the controls are satisfied
            .filter(|ScaledState(basis, _)| {
                use ExtendedBasis::*;
                match basis {
                    Binary(bits) => bits & controls.get_bitstring() == controls.get_bitstring(),
                    Superposition(bases) => {
                        use ExtendedQubitBasis::*;
                        controls
                            .get_indices()
                            .iter()
                            .all(|&i| bases.get(i) == Some(&One))
                    }
                }
            })
            .for_each(|ScaledState(basis, scalar)| {
                let eigenvalue = unconditional_single_qubit_gate(basis, target);
                if let Some(phase) = eigenvalue {
                    *scalar = *scalar * phase;
                }
            });

        result
    }

    fn apply_non_discrete_gate(
        states: impl IntoIterator<Item = ScaledState>,
        target: usize,
        transform: impl Fn(ExtendedBasis, usize) -> [Option<ScaledState>; 2] + 'static,
    ) -> impl Iterator<Item = ScaledState> {
        states
            .into_iter()
            .flat_map(move |ScaledState(basis, scalar)| {
                transform(basis.clone(), target)
                    .iter()
                    .filter_map(|s| {
                        if let Some(actual_value) = s {
                            Some(scalar.clone() * actual_value.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
    }

    pub fn apply_u_gate(
        &self,
        controls: QBits,
        target: usize,
        theta: f32,
        phi: f32,
        lambda: f32,
    ) -> Sum {
        let mut states_to_alter: Sum = Vec::new();
        let mut sum: Sum = self.expand_necessary_controls(controls);
        sum.retain(|ss| {
            let ScaledState(basis, _) = ss;
            use ExtendedBasis::*;
            match basis {
                Binary(bits) => {
                    if bits & controls.get_bitstring() == controls.get_bitstring() {
                        states_to_alter.push(ss.clone());
                        return false;
                    } else {
                        return true;
                    }
                }
                Superposition(bases) => {
                    use ExtendedQubitBasis::*;
                    let all_controls_set = controls
                        .get_indices()
                        .iter()
                        .all(|&i| bases.get(i) == Some(&One));

                    if all_controls_set {
                        states_to_alter.push(ss.clone());
                        return false;
                    } else {
                        return true;
                    }
                }
            }
        });

        let r_z_lambda = move |b, t| ExtendedBasis::p(b, t, lambda as f32);
        let r_y_theta = move |b, t| ExtendedBasis::r_y(b, t, theta as f32);
        let r_z_phi = move |b, t| ExtendedBasis::p(b, t, phi as f32);

        let new_state = Self::apply_non_discrete_gate(states_to_alter, target, r_z_lambda);
        let new_state = Self::apply_non_discrete_gate(new_state, target, r_y_theta);
        let new_state = Self::apply_non_discrete_gate(new_state, target, r_z_phi);
        let new_state = new_state.filter(|s| !s.has_zero_coef());

        sum.extend(new_state);
        sum
    }

    /// Expands only states that are necessary to check the controls,
    /// meaning that states where any control qubit is definite zero will be ignored,
    /// while others will be expanded so checks can be performed on the underlying binary states.
    pub fn expand_necessary_controls(&self, controls: QBits) -> Sum {
        let control_indices = controls.get_indices();
        let expanded: Sum = self
            .sum
            .iter()
            .map(|ScaledState(basis, scalar)| {
                use ExtendedBasis::*;
                match basis {
                    Superposition(bases) => {
                        let any_controls_are_zero = control_indices.iter().any(|&i| {
                            use ExtendedQubitBasis::*;
                            bases.get(i).unwrap_or(&Zero) == &Zero
                        });

                        if any_controls_are_zero {
                            return vec![ScaledState(Superposition(bases.clone()), *scalar)];
                        } else {
                            // A term where the controls are fulfilled is contained within `basis`
                            let mut substates = basis
                                .clone() // Probably expensive clone
                                .expand_qubits(controls);

                            // Distribute the scalar to the expanded states
                            substates
                                .iter_mut()
                                .for_each(|ScaledState(_, inner_scalar)| {
                                    *inner_scalar = *scalar * *inner_scalar;
                                });

                            substates
                        }
                    }
                    Binary(bits) => vec![ScaledState(Binary(*bits), *scalar)],
                }
            })
            .flatten()
            .collect();

        expanded
    }

    /// Returns the all expanded states of the current state,
    /// where any state that contains an extended basis is expanded into the binary states it represents.
    /// All states containing an extended basis (eg, |+⟩, |−⟩, |i⟩, |−i⟩),
    /// will be expanded into the binary states they represent.
    ///
    /// **CAUTION**: There is no deduplication or simplification of the resulting states,
    /// so the same binary state may appear multiple times with different scalars,
    /// and these scalars should be summed to get the actual probability of that binary state.
    fn all_states_unsugared(&self) -> impl Iterator<Item = ScaledState> {
        debug!(
            "All states unsugared: {:?}",
            self.all_scaled_states()
                .flat_map(|s| s.all_inherent_states())
                .collect::<Vec<_>>()
        );

        self.all_scaled_states()
            .flat_map(|s| s.all_inherent_states())
    }

    pub fn calculate_probability_distribution(&self) -> &HashMap<usize, Scalar> {
        self.probability_distribution_cache.get_or_init(|| {
            let mut summed_scalars: HashMap<usize, Scalar> = HashMap::new();
            let non_zero_states = self.all_states_unsugared().filter(|s| !s.has_zero_coef());
            for ScaledState(basis, scalar) in non_zero_states {
                let basis = basis.into_binary(); // Unsugaring guarantees binary state here

                if let Some(old_scalar) = summed_scalars.get(&basis) {
                    // There already was a scalar for this basis
                    let new_scalar = *old_scalar + scalar;
                    if new_scalar.probability() == 0f32 {
                        summed_scalars.remove(&basis);
                    } else {
                        summed_scalars.insert(basis, new_scalar);
                    }
                } else {
                    // This basis has no previous terms
                    summed_scalars.insert(basis, scalar);
                }
            }

            debug!("Probability distribution: {:?}", summed_scalars);
            summed_scalars
        })
    }
}

impl QuantumState for HashMap<usize, Scalar> {
    type BasisValue = Scalar;

    fn collapse(&self) -> usize {
        let mut probability_so_far = 0f32;
        let guess = random::<f32>();
        for (state, scalar) in self {
            probability_so_far += scalar.probability();
            if probability_so_far >= guess {
                return *state;
            }
        }

        panic!(
            "Probabilities did not sum to 1! Total probability: {}",
            probability_so_far
        );
    }

    fn basis_value(&self, basis: usize) -> Self::BasisValue {
        let Some(s) = self.get(&basis) else {
            return Scalar::ZERO;
        };

        *s
    }
}
