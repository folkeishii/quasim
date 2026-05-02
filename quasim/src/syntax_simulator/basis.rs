use core::panic;
use std::{collections::VecDeque, fmt::Debug};

use nalgebra::Complex;

use crate::{
    gate::QBits,
    syntax_simulator::{
        scalar::Scalar,
        state::{ScaledState, Sum},
    },
};

pub type TrueQubitBasis = bool;

#[derive(Copy, Clone)]
pub struct ScaledQubitBasis(TrueQubitBasis, Scalar);

pub type ScaledQubit = [ScaledQubitBasis; 2];

impl Debug for ScaledQubitBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}|{}⟩", self.1, if self.0 { 1 } else { 0 })
    }
}

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
    fn semantic_expressions(&self) -> ScaledQubit {
        use ExtendedQubitBasis::*;
        match self {
            Zero => [
                ScaledQubitBasis(false, Scalar::ONE),
                ScaledQubitBasis(false, Scalar::ZERO),
            ],
            One => [
                ScaledQubitBasis(true, Scalar::ONE),
                ScaledQubitBasis(true, Scalar::ZERO),
            ],
            Plus => [
                ScaledQubitBasis(false, Scalar::FRAC_1_SQRT_2),
                ScaledQubitBasis(true, Scalar::FRAC_1_SQRT_2),
            ],
            Minus => [
                ScaledQubitBasis(false, Scalar::FRAC_1_SQRT_2),
                ScaledQubitBasis(true, -Scalar::FRAC_1_SQRT_2),
            ],
            I => [
                ScaledQubitBasis(false, Scalar::FRAC_1_SQRT_2),
                ScaledQubitBasis(true, Scalar::FRAC_1_SQRT_2 * Scalar::I),
            ],
            MinusI => [
                ScaledQubitBasis(false, Scalar::FRAC_1_SQRT_2),
                ScaledQubitBasis(true, -Scalar::FRAC_1_SQRT_2 * Scalar::I),
            ],
        }
    }

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
            Superposition(bases) => {
                let Some(qubit_basis_to_expand) = bases.get(qubit_to_expand) else {
                    // If the qubit to expand is out of bounds, it is effectively in the |0⟩ state, so there is nothing to expand.
                    return [
                        Some(ScaledState(Self::Superposition(bases), Scalar::ONE)),
                        None,
                    ];
                };

                use ExtendedQubitBasis::*;
                match *qubit_basis_to_expand {
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

                let apply_i: bool = matches!(
                    qubit_basis_to_expand,
                    ExtendedQubitBasis::I | ExtendedQubitBasis::MinusI
                );

                let mut zero_case = bases.clone();
                zero_case[qubit_to_expand] = Zero;
                let mut one_case = bases;
                one_case[qubit_to_expand] = One;
                let one_scalar = if apply_i {
                    Scalar::I
                        * if apply_minus {
                            -Scalar::FRAC_1_SQRT_2
                        } else {
                            Scalar::FRAC_1_SQRT_2
                        }
                } else {
                    if apply_minus {
                        -Scalar::FRAC_1_SQRT_2
                    } else {
                        Scalar::FRAC_1_SQRT_2
                    }
                };

                [
                    Some(ScaledState(
                        Self::Superposition(zero_case),
                        Scalar::FRAC_1_SQRT_2,
                    )),
                    Some(ScaledState(Self::Superposition(one_case), one_scalar)),
                ]
            }
        }
    }

    /// Expands all the qubit indexes in the given list, meaning that qubits in an extended
    /// basis (eg, |+⟩, |−⟩, |i⟩, |−i⟩) will be expanded into a sum of binary states,
    /// while qubits already in a binary basis (eg, |0⟩, |1⟩) will be left unchanged.
    pub fn expand_qubits(self, qubits_to_expand: QBits) -> Sum {
        self.expand_qubits_helper(qubits_to_expand.get_indices().into())
    }

    fn expand_qubits_helper(self, mut qubits_to_expand: VecDeque<usize>) -> Sum {
        let Some(qubit_to_expand) = qubits_to_expand.pop_front() else {
            return vec![ScaledState(self, Scalar::ONE)];
        };

        use ExtendedBasis::*;
        match self {
            Binary(_) => vec![ScaledState(self, Scalar::ONE)],
            Superposition(_) => {
                // Expand the first qubit, which yields just two terms.
                let possible_expansion = self.clone().expand_qubit(qubit_to_expand);

                // Actually, expanding |0⟩ or |1⟩ just gives one term, so we need to filter out the None case.
                let just_existing_expansions = possible_expansion
                    .iter()
                    // Flatten removes the Option wrapping
                    .flatten();

                // At this point we have all the expansions for the first qubit to expand.
                // Take those terms we expanded into, and just run this function on those terms,
                // but now with one less qubit to expand.

                let all_terms = just_existing_expansions
                    .flat_map(|ScaledState(basis, scalar)| {
                        // If a scaled state turns into more terms, the scalar needs to be distributed across the terms.
                        *scalar * basis.clone().expand_qubits_helper(qubits_to_expand.clone()) // Expensive clone(s)?
                    })
                    .collect::<Sum>();

                all_terms
            }
        }
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

    // This function essentially does the same as expand_qubits, but it always
    // results in binary states. There are assumptions elsewhere that this function
    // only returns binary states, and it implicitly only ever returns
    // the ExtendedBasis::Binary variant.
    pub fn all_inherent_states(self) -> Sum {
        let mut extended_basis = match self {
            ExtendedBasis::Binary(bitstring) => {
                return vec![ScaledState(ExtendedBasis::Binary(bitstring), Scalar::ONE)];
            }
            ExtendedBasis::Superposition(b) => b,
        };

        let Some(qubit_basis) = extended_basis.pop() else {
            panic!("Cannot get inherent states of empty basis!");
        };

        if extended_basis.is_empty() {
            return qubit_basis
                .semantic_expressions()
                .iter()
                .filter(|ScaledQubitBasis(_, scalar)| !scalar.is_zero())
                .map(|ScaledQubitBasis(basis, scalar)| {
                    let basis = if *basis { 1 } else { 0 };
                    ScaledState(ExtendedBasis::Binary(basis), *scalar)
                })
                .collect();
        }

        let semantics = qubit_basis.semantic_expressions();

        semantics
            .into_iter()
            .filter(|ScaledQubitBasis(_, s)| !s.is_zero())
            .map(|msb| {
                let lesser_bits =
                    ExtendedBasis::Superposition(extended_basis.clone()).all_inherent_states();
                lesser_bits
                    .iter()
                    .map(|lsb| {
                        let lsb_state = match &lsb.0 {
                            ExtendedBasis::Binary(bitstring) => bitstring,
                            ExtendedBasis::Superposition(_) => {
                                panic!("Lesser bits should always be binary at this point!")
                            }
                        };
                        let basis = (msb.0 as usize) << extended_basis.len() | lsb_state;
                        let scalar_product = msb.1 * lsb.1;
                        ScaledState(ExtendedBasis::Binary(basis), scalar_product)
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
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
        let eigenvalue;

        use ExtendedBasis::*;
        match self {
            Binary(bits) => {
                let bit_is_one = (*bits & (1 << target)) != 0;
                *bits = *bits ^ (1 << target);
                eigenvalue = Some(if bit_is_one { -Scalar::I } else { Scalar::I });
            }
            Superposition(bases) => {
                Self::pad_to_length(bases, target + 1);
                eigenvalue = match bases[target] {
                    ExtendedQubitBasis::Zero => Some(Scalar::I),
                    ExtendedQubitBasis::One => Some(-Scalar::I),
                    ExtendedQubitBasis::Plus => Some(-Scalar::I),
                    ExtendedQubitBasis::Minus => Some(Scalar::I),
                    ExtendedQubitBasis::I => None,
                    ExtendedQubitBasis::MinusI => Some(-Scalar::ONE),
                };
                bases[target] = bases[target].y();
            }
        }
        eigenvalue
    }

    pub fn z(&mut self, target: usize) -> Option<Scalar> {
        let mut eigenvalue = None;

        use ExtendedBasis::*;
        match self {
            Binary(bits) => {
                if target < usize::BITS as usize && (*bits & (1 << target)) != 0 {
                    eigenvalue = Some(-Scalar::ONE);
                }
            }
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
            Binary(bits) => {
                if target < usize::BITS as usize && (*bits & (1 << target)) != 0 {
                    eigenvalue = Some(Scalar::I);
                }
            }
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
                let half_angle = Complex::from(angle / 2f32);
                let expanded = self.expand_qubit(target);

                let Some(ScaledState(ref guaranteed_basis, ref guaranteed_scalar)) = expanded[0]
                else {
                    panic!("Would expect at least one state after expansion");
                };

                let top_left = Scalar::from(half_angle.cos());
                let bottom_left = Scalar::from(half_angle.sin());
                let top_right = Scalar::from(-half_angle.sin());
                let bottom_right = top_left;
                let r: [Option<ScaledState>; 2] =
                    if let Some(ScaledState(ref one_basis, ref one_scalar)) = expanded[1] {
                        // If there are two states after expansion, guaranteed_basis is the |0⟩ case and one_basis is the |1⟩ case
                        let zero_basis = guaranteed_basis;
                        let zero_scalar = guaranteed_scalar;
                        let top_left = top_left * *zero_scalar;
                        let bottom_left = bottom_left * *zero_scalar;
                        let top_right = top_right * *one_scalar;
                        let bottom_right = bottom_right * *one_scalar;

                        [
                            Some(ScaledState(zero_basis.clone(), top_left + top_right)),
                            Some(ScaledState(one_basis.clone(), bottom_left + bottom_right)),
                        ]
                    } else {
                        // Zero case only expands into one state, so the target is already binary
                        let guaranteed_basis_vec = guaranteed_basis.clone().vec_form();
                        use ExtendedQubitBasis::*;
                        let old_target_basis = match guaranteed_basis_vec.get(target) {
                            None => {
                                // If the target qubit is out of bounds, it is effectively in the |0⟩ state
                                &Zero
                            }
                            Some(basis) => basis,
                        };

                        match old_target_basis {
                            Zero => {
                                // |0⟩ case only expands into |0⟩, so the target is in the |0⟩ state
                                // After R_y, the |0⟩ state becomes cos(θ/2)|0⟩ + sin(θ/2)|1⟩

                                let zero_basis = guaranteed_basis.clone();
                                let mut one_basis = guaranteed_basis.clone();
                                one_basis.x(target);

                                [
                                    Some(*guaranteed_scalar * ScaledState(zero_basis, top_left)),
                                    Some(*guaranteed_scalar * ScaledState(one_basis, bottom_left)),
                                ]
                            }
                            One => {
                                // |1⟩ case only expands into |1⟩, so the target is in the |1⟩ state
                                // After R_y, the |1⟩ state becomes -sin(θ/2)|0⟩ + cos(θ/2)|1⟩

                                let mut zero_basis = guaranteed_basis.clone();
                                zero_basis.x(target);
                                let one_basis = guaranteed_basis.clone();

                                [
                                    Some(*guaranteed_scalar * ScaledState(zero_basis, top_right)),
                                    Some(*guaranteed_scalar * ScaledState(one_basis, bottom_right)),
                                ]
                            }
                            _ => panic!(
                                "Target qubit should have already been expanded into binary!"
                            ),
                        }
                    };

                r
            }
        }
    }

    pub fn p(self, target: usize, angle: f32) -> [Option<ScaledState>; 2] {
        use ExtendedBasis::*;
        match self {
            Binary(bits) => {
                let phase_change: Scalar = Complex::exp(Complex::I * angle).into();
                let scalar = if target < usize::BITS as usize && (bits & (1 << target)) != 0 {
                    phase_change
                } else {
                    Scalar::ONE
                };
                [Some(ScaledState(Binary(bits), scalar)), None]
            }
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

impl From<ScaledQubitBasis> for ScaledState {
    fn from(value: ScaledQubitBasis) -> Self {
        let basis = match value.0 {
            false => 0,
            true => 1,
        };
        ScaledState(ExtendedBasis::Binary(basis), value.1)
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
