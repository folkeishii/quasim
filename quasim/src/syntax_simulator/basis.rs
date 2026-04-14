use std::fmt::Debug;

use nalgebra::Complex;

use crate::{
    gate::QBits,
    syntax_simulator::{ScaledState, scalar::Scalar, state::Sum},
};

#[derive(Clone, Copy, PartialEq)]
pub enum ExtendedQubitBasis {
    Zero,
    One,
    Plus,
    Minus,
    I,
    MinusI,
}

impl Debug for ExtendedQubitBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ExtendedQubitBasis::*;
        let s = match self {
            Zero => "0",
            One => "1",
            Plus => "+",
            Minus => "-",
            I => "i",
            MinusI => "(-i)",
        };
        write!(f, "{}", s)
    }
}

impl ExtendedQubitBasis {
    // Gates

    #[inline(always)]
    fn x(&self) -> ExtendedQubitBasis {
        use ExtendedQubitBasis::*;
        match self {
            Zero => One,
            One => Zero,
            Plus => Plus,
            Minus => Minus,
            I => MinusI,
            MinusI => I,
        }
    }

    #[inline(always)]
    fn y(&self) -> ExtendedQubitBasis {
        use ExtendedQubitBasis::*;
        match self {
            Zero => One,
            One => Zero,
            Plus => Minus,
            Minus => Plus,
            I => I,
            MinusI => MinusI,
        }
    }

    fn z(&self) -> ExtendedQubitBasis {
        use ExtendedQubitBasis::*;
        match self {
            Zero => Zero,
            One => One,
            Plus => Minus,
            Minus => Plus,
            I => MinusI,
            MinusI => I,
        }
    }

    #[inline(always)]
    fn h(&self) -> ExtendedQubitBasis {
        use ExtendedQubitBasis::*;
        match self {
            Zero => Plus,
            One => Minus,
            Plus => Zero,
            Minus => One,
            I => MinusI,
            MinusI => I,
        }
    }

    #[inline(always)]
    fn s(&self) -> ExtendedQubitBasis {
        // The S gate is just a 90 degree rotation around z-axis on bloch sphere.
        use ExtendedQubitBasis::*;
        match self {
            Zero => Zero,
            One => One,
            Plus => I,
            I => Minus,
            Minus => MinusI,
            MinusI => Plus,
        }
    }
}

#[derive(Clone)]
pub enum ExtendedBasis {
    Binary(usize),
    Superposition(Vec<ExtendedQubitBasis>),
}

impl ExtendedBasis {
    /// Expands the given qubit in the basis, returning the two cases of the expansion if
    /// the qubit is in an extended basis (eg, |+⟩, |−⟩, |i⟩, |−i⟩),
    /// otherwise returning the same basis if the qubit is already binary (eg, |0⟩, |1⟩).
    pub fn expand_qubit(self, qubit_to_expand: usize) -> [Option<ScaledState>; 2] {
        use ExtendedBasis::*;
        match self {
            Binary(_) => [Some(ScaledState(self, Scalar::ONE)), None],
            Superposition(mut bases) => {
                Self::pad_to_length(&mut bases, qubit_to_expand + 1);
                let qubit_basis_to_expand = bases[qubit_to_expand];

                use ExtendedQubitBasis::*;
                match qubit_basis_to_expand {
                    One | Zero => {
                        // If the qubit is already in a binary state, there is nothing to expand.
                        return [
                            Some(ScaledState(Self::Superposition(bases), Scalar::ONE)),
                            None,
                        ];
                    }
                    _ => (),
                };

                let apply_minus: bool = matches!(
                    qubit_basis_to_expand,
                    ExtendedQubitBasis::Minus | ExtendedQubitBasis::MinusI
                );

                let mut zero_case = bases.clone();
                zero_case[qubit_to_expand] = Zero;
                let mut one_case = bases;
                one_case[qubit_to_expand] = One;

                [
                    Some(ScaledState(
                        Self::Superposition(zero_case),
                        Scalar::FRAC_1_SQRT_2,
                    )),
                    Some(ScaledState(
                        Self::Superposition(one_case),
                        if apply_minus {
                            -Scalar::FRAC_1_SQRT_2
                        } else {
                            Scalar::FRAC_1_SQRT_2
                        },
                    )),
                ]
            }
        }
    }

    /// Expands all the qubit indexes in the given list, meaning that qubits in an extended
    /// basis (eg, |+⟩, |−⟩, |i⟩, |−i⟩) will be expanded into a sum of binary states,
    /// while qubits already in a binary basis (eg, |0⟩, |1⟩) will be left unchanged.
    pub fn expand_qubits(self, qubits_to_expand: QBits) -> Sum {
        let mut states = vec![ScaledState(self, Scalar::ONE)];

        for qubit_to_expand in qubits_to_expand.get_indices() {
            let mut next = Vec::with_capacity(states.len() * 2);

            for ScaledState(basis, scalar) in states {
                use ExtendedBasis::*;
                match basis {
                    Binary(bits) => next.push(ScaledState(Binary(bits), scalar)),
                    Superposition(mut bases) => {
                        let qubit_basis_to_expand = bases
                            .get(qubit_to_expand)
                            .copied()
                            .unwrap_or(ExtendedQubitBasis::Zero);

                        use ExtendedQubitBasis::*;
                        match qubit_basis_to_expand {
                            Zero | One => {
                                next.push(ScaledState(Superposition(bases), scalar));
                            }
                            Plus => {
                                Self::pad_to_length(&mut bases, qubit_to_expand + 1);
                                let mut zero_case = bases.clone();
                                zero_case[qubit_to_expand] = Zero;
                                bases[qubit_to_expand] = One;

                                next.push(ScaledState(
                                    Superposition(zero_case),
                                    scalar * Scalar::FRAC_1_SQRT_2,
                                ));
                                next.push(ScaledState(
                                    Superposition(bases),
                                    scalar * Scalar::FRAC_1_SQRT_2,
                                ));
                            }
                            Minus => {
                                Self::pad_to_length(&mut bases, qubit_to_expand + 1);
                                let mut zero_case = bases.clone();
                                zero_case[qubit_to_expand] = Zero;
                                bases[qubit_to_expand] = One;

                                next.push(ScaledState(
                                    Superposition(zero_case),
                                    scalar * Scalar::FRAC_1_SQRT_2,
                                ));
                                next.push(ScaledState(
                                    Superposition(bases),
                                    scalar * -Scalar::FRAC_1_SQRT_2,
                                ));
                            }
                            I => {
                                Self::pad_to_length(&mut bases, qubit_to_expand + 1);
                                let mut zero_case = bases.clone();
                                zero_case[qubit_to_expand] = Zero;
                                bases[qubit_to_expand] = One;

                                next.push(ScaledState(
                                    Superposition(zero_case),
                                    scalar * Scalar::FRAC_1_SQRT_2,
                                ));
                                next.push(ScaledState(
                                    Superposition(bases),
                                    scalar * (Scalar::FRAC_1_SQRT_2 * Scalar::I),
                                ));
                            }
                            MinusI => {
                                Self::pad_to_length(&mut bases, qubit_to_expand + 1);
                                let mut zero_case = bases.clone();
                                zero_case[qubit_to_expand] = Zero;
                                bases[qubit_to_expand] = One;

                                next.push(ScaledState(
                                    Superposition(zero_case),
                                    scalar * Scalar::FRAC_1_SQRT_2,
                                ));
                                next.push(ScaledState(
                                    Superposition(bases),
                                    scalar * (-Scalar::FRAC_1_SQRT_2 * Scalar::I),
                                ));
                            }
                        }
                    }
                }
            }

            states = next;
        }

        states
    }

    pub fn try_into_binary(&self) -> Option<usize> {
        match self {
            ExtendedBasis::Binary(bitstring) => Some(*bitstring),
            ExtendedBasis::Superposition(_) => None,
        }
    }

    pub fn into_binary(&self) -> usize {
        self.try_into_binary()
            .expect("Cannot convert superposition basis into binary basis!")
    }

    fn vec_form(self) -> Vec<ExtendedQubitBasis> {
        match self {
            ExtendedBasis::Binary(mut bitstring) => {
                let mut bases = Vec::new();
                while bitstring != 0 {
                    let bit_is_one = (bitstring & 1) == 1;
                    bases.push(if bit_is_one {
                        ExtendedQubitBasis::One
                    } else {
                        ExtendedQubitBasis::Zero
                    });
                    bitstring = bitstring >> 1;
                }
                bases
            }
            ExtendedBasis::Superposition(bases) => bases,
        }
    }

    fn vec_form_padded_to_len(self, n: usize) -> Vec<ExtendedQubitBasis> {
        let mut vec_form = self.vec_form();
        if vec_form.len() < n {
            vec_form.resize(n, ExtendedQubitBasis::Zero);
        }
        vec_form
    }

    fn pad_to_length(superposition: &mut Vec<ExtendedQubitBasis>, n: usize) {
        if superposition.len() < n {
            superposition.resize(n, ExtendedQubitBasis::Zero);
        }
    }

    pub fn all_inherent_states(self: ExtendedBasis) -> Sum {
        match self {
            Self::Binary(bitstring) => vec![ScaledState(Self::Binary(bitstring), Scalar::ONE)],
            Self::Superposition(bases) => {
                let mut expanded = vec![(0usize, Scalar::ONE)];

                for (qubit_index, qubit_basis) in bases.into_iter().enumerate() {
                    use ExtendedQubitBasis::*;
                    let mut next = Vec::with_capacity(expanded.len() * 2);
                    for (bits, scalar) in expanded {
                        match qubit_basis {
                            Zero => next.push((bits, scalar)),
                            One => next.push((bits | (1 << qubit_index), scalar)),
                            Plus => {
                                next.push((bits, scalar * Scalar::FRAC_1_SQRT_2));
                                next.push((
                                    bits | (1 << qubit_index),
                                    scalar * Scalar::FRAC_1_SQRT_2,
                                ));
                            }
                            Minus => {
                                next.push((bits, scalar * Scalar::FRAC_1_SQRT_2));
                                next.push((
                                    bits | (1 << qubit_index),
                                    scalar * -Scalar::FRAC_1_SQRT_2,
                                ));
                            }
                            I => {
                                next.push((bits, scalar * Scalar::FRAC_1_SQRT_2));
                                next.push((
                                    bits | (1 << qubit_index),
                                    scalar * (Scalar::FRAC_1_SQRT_2 * Scalar::I),
                                ));
                            }
                            MinusI => {
                                next.push((bits, scalar * Scalar::FRAC_1_SQRT_2));
                                next.push((
                                    bits | (1 << qubit_index),
                                    scalar * (-Scalar::FRAC_1_SQRT_2 * Scalar::I),
                                ));
                            }
                        }
                    }
                    expanded = next;
                }

                expanded
                    .into_iter()
                    .filter(|(_, scalar)| !scalar.is_zero())
                    .map(|(bits, scalar)| ScaledState(Self::Binary(bits), scalar))
                    .collect()
            }
        }
    }

    // Gates

    pub fn x(&mut self, target: usize) -> Option<Scalar> {
        let mut eigenvalue = None;

        use ExtendedBasis::*;
        match self {
            Binary(bits) => *bits = *bits ^ (1 << target),
            Superposition(bases) => {
                Self::pad_to_length(bases, target + 1);
                if bases[target] == ExtendedQubitBasis::Minus {
                    eigenvalue = Some(-Scalar::ONE);
                }
                bases[target] = bases[target].x();
            }
        }
        eigenvalue
    }

    pub fn y(&mut self, target: usize) -> Option<Scalar> {
        let mut eigenvalue = None;

        use ExtendedBasis::*;
        match self {
            Binary(bits) => *bits = *bits ^ (1 << target),
            Superposition(bases) => {
                Self::pad_to_length(bases, target + 1);
                if bases[target] == ExtendedQubitBasis::MinusI {
                    eigenvalue = Some(-Scalar::ONE);
                }
                bases[target] = bases[target].y();
            }
        }
        eigenvalue
    }

    pub fn z(&mut self, target: usize) -> Option<Scalar> {
        let mut eigenvalue = None;

        use ExtendedBasis::*;
        match self {
            Binary(_) => (),
            Superposition(bases) => {
                Self::pad_to_length(bases, target + 1);
                if bases[target] == ExtendedQubitBasis::One {
                    eigenvalue = Some(-Scalar::ONE);
                }
                bases[target] = bases[target].z();
            }
        }
        eigenvalue
    }

    pub fn h(&mut self, target: usize) -> Option<Scalar> {
        use ExtendedBasis::*;
        match self {
            Binary(bits) => {
                let bit_is_one = (*bits & (1 << target)) != 0;
                let new_qubit_basis = if bit_is_one {
                    ExtendedQubitBasis::Minus
                } else {
                    ExtendedQubitBasis::Plus
                };
                // Clone is cheap because self is effectively a usize
                let mut new_bases = self.clone().vec_form_padded_to_len(target + 1);
                new_bases[target] = new_qubit_basis;
                *self = Superposition(new_bases);
            }
            Superposition(bases) => {
                Self::pad_to_length(bases, target + 1);
                bases[target] = bases[target].h();
            }
        }
        None // This simulator does not have syntax for representing the eigenvector of the H gate
    }

    pub fn s(&mut self, target: usize) -> Option<Scalar> {
        let mut eigenvalue = None;

        use ExtendedBasis::*;
        match self {
            Binary(_) => (),
            Superposition(bases) => {
                Self::pad_to_length(bases, target + 1);
                if bases[target] == ExtendedQubitBasis::One {
                    eigenvalue = Some(Scalar::I);
                }
                bases[target] = bases[target].s();
            }
        }
        eigenvalue
    }

    pub fn r_y(self, target: usize, angle: f32) -> [Option<ScaledState>; 2] {
        use ExtendedBasis::*;
        match self {
            Binary(_) => Self::r_y(
                Superposition(self.vec_form_padded_to_len(target + 1)),
                target,
                angle,
            ),
            Superposition(_) => {
                let angle = Complex::from(angle / 2f32);
                let [zero_case, one_case] = self.expand_qubit(target);

                let Some(ScaledState(zero_basis, zero_scalar)) = zero_case else {
                    panic!("Would expect at least one state after expansion");
                };

                let top_left = Scalar::from(angle.cos()) * zero_scalar;
                let bottom_left = Scalar::from(angle.sin()) * zero_scalar;

                let r: [Option<ScaledState>; 2] =
                    if let Some(ScaledState(one_basis, one_scalar)) = one_case {
                        let top_right = Scalar::from(-angle.sin()) * one_scalar;
                        let bottom_right = Scalar::from(angle.cos()) * one_scalar;

                        [
                            Some(ScaledState(zero_basis, top_left + top_right)),
                            Some(ScaledState(one_basis, bottom_left + bottom_right)),
                        ]
                    } else {
                        // Zero case only expands into one state, so the target is already binary
                        let mut one_basis = zero_basis.clone();
                        one_basis.x(target);

                        [
                            // Top right and bottom right are zero because there is no one case
                            Some(ScaledState(zero_basis, top_left)),
                            Some(ScaledState(one_basis, bottom_left)),
                        ]
                    };

                r
            }
        }
    }

    pub fn p(self, target: usize, angle: f32) -> [Option<ScaledState>; 2] {
        use ExtendedBasis::*;
        match self {
            Binary(_) => [Some(ScaledState(self, Scalar::ONE)), None],
            Superposition(_) => {
                let mut result = self.expand_qubit(target);
                let phase_change: Scalar = Complex::exp(Complex::I * angle).into();

                result.iter_mut().flatten().for_each(|ScaledState(basis, scalar)| {
                    use ExtendedBasis::*;
                    match basis {
                        Binary(bits) => {
                            if (*bits & (1 << target)) != 0 {
                                *scalar = *scalar * phase_change;
                            }
                        }
                        Superposition(bases) => {
                            if let Some(qubit_basis) = bases.get(target) {
                                use ExtendedQubitBasis::*;
                                match qubit_basis {
                                    One => *scalar = *scalar * phase_change,
                                    Zero => (),
                                    _ => panic!("Target qubit should have already been expanded into binary!"),
                                }
                            } else {
                                // If the target qubit is out of bounds, it is effectively in the |0⟩ state, so no phase change should be applied.
                            }
                        }
                    }
                });

                result
            }
        }
    }
}

impl Debug for ExtendedBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtendedBasis::Binary(bitstring) => write!(f, "|{}⟩", bitstring),
            ExtendedBasis::Superposition(bases) => {
                write!(f, "|")?;
                for basis in bases.iter().rev() {
                    write!(f, "{:?}", basis)?;
                }
                write!(f, "⟩")?;
                Ok(())
            }
        }
    }
}

impl PartialEq for ExtendedBasis {
    fn eq(&self, other: &Self) -> bool {
        use ExtendedBasis::*;
        match (self, other) {
            (Binary(b1), Binary(b2)) => b1 == b2,
            (Superposition(s1), Superposition(s2)) => s1 == s2,
            (Binary(l), Superposition(r)) => {
                let mut l = *l;
                for basis in r {
                    let bit_is_one = (l & 1) == 1;
                    let expected_basis = if bit_is_one {
                        ExtendedQubitBasis::One
                    } else {
                        ExtendedQubitBasis::Zero
                    };
                    if *basis != expected_basis {
                        return false;
                    }
                    l = l >> 1;
                }
                true
            }
            _ => other == self,
        }
    }
}
