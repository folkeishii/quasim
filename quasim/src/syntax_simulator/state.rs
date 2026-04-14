use std::fmt::Debug;

use log::debug;

use crate::{
    gate::{Gate, GateType, QBits},
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
            .into_iter()
            .map(|substate| my_scalar * substate)
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
}

impl SumOfScaledStates {
    pub fn guaranteed_full_zero(n_qubits: usize) -> Self {
        let n_zeros = ScaledState(ExtendedBasis::Binary(0), Scalar::ONE);
        Self {
            n_qubits,
            sum: vec![n_zeros],
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
        let controls = gate.get_control_bits();
        let targets = gate.get_target_bits().get_indices(); // O(n)

        debug!(
            "Applying gate {:?} with controls {:?} and target {:?}",
            gate.get_type(),
            controls,
            targets
        );

        // O(3 X applications...)
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

        use GateType::*;
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
        let control_indices = controls.get_indices();
        let control_bitstring = controls.get_bitstring();
        let mut result = self.expand_necessary_controls(controls);

        result
            .iter_mut()
            // Keep only the states where all the controls are satisfied
            .filter(|ScaledState(basis, _)| {
                use ExtendedBasis::*;
                match basis {
                    Binary(bits) => bits & control_bitstring == control_bitstring,
                    Superposition(bases) => {
                        use ExtendedQubitBasis::*;
                        control_indices.iter().all(|&i| bases.get(i) == Some(&One))
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
                transform(basis, target)
                    .into_iter()
                    .flatten()
                    .map(move |actual_value| scalar * actual_value)
            })
    }

    pub fn apply_u_gate(
        &self,
        controls: QBits,
        target: usize,
        theta: f64,
        phi: f64,
        lambda: f64,
    ) -> Sum {
        let sum: Sum = self.expand_necessary_controls(controls);
        let mut states_to_alter: Sum = Vec::with_capacity(sum.len());
        let mut passthrough: Sum = Vec::with_capacity(sum.len());
        let control_indices = controls.get_indices();
        let control_bitstring = controls.get_bitstring();

        for ss in sum {
            let should_alter = {
                let ScaledState(basis, _) = &ss;
                use ExtendedBasis::*;
                match basis {
                    Binary(bits) => bits & control_bitstring == control_bitstring,
                    Superposition(bases) => {
                        use ExtendedQubitBasis::*;
                        control_indices.iter().all(|&i| bases.get(i) == Some(&One))
                    }
                }
            };

            if should_alter {
                states_to_alter.push(ss);
            } else {
                passthrough.push(ss);
            }
        }

        let r_z_lambda = move |b, t| ExtendedBasis::p(b, t, lambda as f32);
        let r_y_theta = move |b, t| ExtendedBasis::r_y(b, t, theta as f32);
        let r_z_phi = move |b, t| ExtendedBasis::p(b, t, phi as f32);

        let new_state = Self::apply_non_discrete_gate(states_to_alter, target, r_z_lambda);
        let new_state = Self::apply_non_discrete_gate(new_state, target, r_y_theta);
        let new_state = Self::apply_non_discrete_gate(new_state, target, r_z_phi);
        let new_state = new_state.filter(|s| !s.has_zero_coef());

        passthrough.extend(new_state);
        passthrough
    }

    /// Expands only states that are necessary to check the controls,
    /// meaning that states where any control qubit is definite zero will be ignored,
    /// while others will be expanded so checks can be performed on the underlying binary states.
    pub fn expand_necessary_controls(&self, controls: QBits) -> Sum {
        let control_indices = controls.get_indices();
        let mut expanded: Sum = Vec::with_capacity(self.sum.len());

        for ScaledState(basis, scalar) in &self.sum {
            use ExtendedBasis::*;
            match basis {
                Superposition(bases) => {
                    // O(n) (control indices)
                    let any_controls_are_zero = control_indices.iter().any(|&i| {
                        use ExtendedQubitBasis::*;
                        bases.get(i).unwrap_or(&Zero) == &Zero
                    });

                    if any_controls_are_zero {
                        expanded.push(ScaledState(Superposition(bases.clone()), *scalar));
                    } else {
                        // A term where the controls are fulfilled is contained within `basis`
                        let mut substates = basis.clone().expand_qubits(controls);

                        // Distribute the scalar to the expanded states
                        for ScaledState(_, inner_scalar) in &mut substates {
                            *inner_scalar = *scalar * *inner_scalar;
                        }

                        expanded.extend(substates);
                    }
                }
                Binary(bits) => expanded.push(ScaledState(Binary(*bits), *scalar)),
            }
        }

        expanded
    }
}
