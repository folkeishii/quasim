use log::trace;

use crate::{
    gate::{Gate, GateType, QBits},
    syntax_simulator::{
        basis::{ExtendedBasis, ExtendedQubitBasis},
        scalar::Scalar,
    },
};

#[derive(Debug, Clone, PartialEq)]
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
}

impl From<(usize, Scalar)> for ScaledState {
    fn from(value: (usize, Scalar)) -> Self {
        let basis = value.0;
        let scalar = value.1;
        ScaledState(ExtendedBasis::Binary(basis), scalar)
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

    /// Returns the probability distribution over all basis states
    /// that exist at this point. Extended bases (eg, |+⟩, |−⟩, |i⟩, |−i⟩)
    /// will not be expanded, so the resulting states may still contain extended bases.
    /// If you want to get the distribution over only binary states,
    /// use `unsugared_probability_distribution` instead.
    pub fn probability_distribution(&self) -> impl Iterator<Item = &ScaledState> {
        self.sum.iter()
    }

    pub fn apply_gate(&mut self, gate: &Gate) {
        let controls = gate.get_control_bits();
        let targets = gate.get_target_bits();
        assert_eq!(targets.get_indices().len(), 1);
        let target = targets.get_indices()[0];

        trace!(
            "Applying gate {:?} with controls {:?} and target {}",
            gate.get_type(),
            controls,
            target
        );

        use GateType::*;
        self.sum = match gate.get_type() {
            X => self.apply_controlled_gate(ExtendedBasis::x, controls, target),
            Y => self.apply_controlled_gate(ExtendedBasis::y, controls, target),
            Z => self.apply_controlled_gate(ExtendedBasis::z, controls, target),
            H => self.apply_controlled_gate(ExtendedBasis::h, controls, target),
            S => self.apply_controlled_gate(ExtendedBasis::s, controls, target),
            SWAP => {
                panic!("SWAP gates should have been removed in the circuit preprocessing step!")
            }
            U(theta, phi, lambda) => self.apply_u_gate(controls, target, theta, phi, lambda),
        };
    }

    pub fn apply_controlled_gate(
        &self,
        unconditional_single_qubit_gate: fn(&mut ExtendedBasis, usize),
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
            .for_each(|ScaledState(basis, _)| {
                unconditional_single_qubit_gate(basis, target);
            });

        result
    }

    pub fn apply_u_gate(
        &self,
        controls: QBits,
        target: usize,
        theta: f64,
        phi: f64,
        lambda: f64,
    ) -> Sum {
        let is_just_phase_change = theta == 0f64;
        let mut states_to_alter: Vec<ScaledState> = Vec::new();
        let mut sum = self.expand_necessary_controls(controls);
        sum.retain(|ss| {
            let ScaledState(basis, _) = ss;
            use ExtendedBasis::*;
            match basis {
                Binary(bits) => bits & controls.get_bitstring() == controls.get_bitstring(),
                Superposition(bases) => {
                    if is_just_phase_change && matches!(bases[target], Zero | One) {
                        // The zero and one states are unaffected by phase changes. Just retain the state.
                        return true;
                    }

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

        let new_states = states_to_alter
            .iter()
            .flat_map(|ScaledState(basis, _)| basis.clone().r_z(target, lambda as f32))
            .flatten()
            .map(|ss| ss.clone())
            .collect::<Vec<_>>();

        let new_states = new_states
            .iter()
            .flat_map(|ScaledState(basis, _)| basis.clone().r_y(target, theta as f32))
            .map(|ss| ss.clone())
            .collect::<Vec<_>>();

        let new_states = new_states
            .iter()
            .flat_map(|ScaledState(basis, _)| basis.clone().r_z(target, phi as f32))
            .flatten()
            .map(|ss| ss.clone())
            .collect::<Vec<_>>();

        sum.extend(new_states);
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
                            basis.clone().expand_qubits(controls) // Probably expensive clone
                        }
                    }
                    Binary(bits) => vec![ScaledState(Binary(*bits), *scalar)],
                }
            })
            .flatten()
            .collect();

        expanded
    }
}
