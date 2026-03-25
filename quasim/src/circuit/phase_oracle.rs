use crate::circuit::{Circuit, CircuitBehaviour};

/// n is the number of classical bits.
pub fn fn_to_truth_table(f: &dyn Fn(&usize) -> bool, n: usize) -> Vec<bool> {
    let mut truth_table = Vec::with_capacity(1 << n);
    for i in 0..(1 << n) {
        truth_table.push(f(&i));
    }
    return truth_table;
}

/// In-place transformation of the truth table to get the ANF coefficients.
pub fn truth_table_to_anf_coefs(mut truth_table: Vec<bool>) -> Vec<bool> {
    let size = truth_table.len();
    let mut step = 1;

    while step < size {
        for i in (0..size).step_by(2 * step) {
            for j in i..(i + step).min(size) {
                if j + step < size {
                    truth_table[j + step] ^= truth_table[j];
                }
            }
        }
        step *= 2;
    }
    truth_table
}

/// Converts the ANF coefficients back to a function that can be evaluated on inputs.
fn anf_coefs_to_fn(anf_coefs: Vec<bool>) -> Box<dyn Fn(&usize) -> bool> {
    let f = move |x: &usize| {
        let mut classical_result = false;

        for (i, coef) in anf_coefs.iter().enumerate() {
            if !coef {
                continue; // Skip if the coefficient is not part of the expression
            }

            if x & i == i || i == 0 {
                classical_result = !classical_result;
            }
        }

        return classical_result;
    };

    return Box::new(f);
}

fn controls_from_bit_mask(possible_controls: &[usize], mut bit_mask: usize) -> Vec<usize> {
    let mut bit_vec = Vec::new();
    let mut i = 0;
    while bit_mask != 0 {
        if bit_mask & 1 == 1 {
            bit_vec.push(possible_controls[i]);
        }
        bit_mask >>= 1;
        i += 1;
    }
    return bit_vec;
}

/// Appends a phase oracle to the given circuit based on the provided ANF coefficients.
pub fn append_phase_oracle<B: CircuitBehaviour>(
    mut circuit: Circuit<B>,
    input_qubits: Vec<usize>,
    target: usize,
    anf_coefs: Vec<bool>,
) -> Circuit<B> {
    for (i, coef) in anf_coefs.iter().enumerate() {
        if !coef {
            continue; // Skip if the coefficient is not part of the expression
        }

        if i == 0 {
            // Apply Z gate for the constant term
            circuit = circuit.z(target);
            continue;
        }

        // Apply a multi-controlled Z gate for the term corresponding to index i
        let control_qubits = controls_from_bit_mask(&input_qubits, i);
        circuit = circuit.cz(&control_qubits, target);
    }

    circuit
}

#[cfg(test)]
mod anf_conversion_tests {

    use super::*;

    fn test_isomorphism(name: &str, f: &dyn Fn(&usize) -> bool) {
        const N: usize = 12; // Number of bits for the truth table

        let truth_table = fn_to_truth_table(&f, N);
        println!("Truth Table for {}: {:?}", name, truth_table);
        let coefs = truth_table_to_anf_coefs(truth_table);
        println!("ANF Coefficients for {}: {:?}", name, coefs);
        let f_reconstructed = anf_coefs_to_fn(coefs);
        for i in 0..(1 << N) {
            assert_eq!(
                f(&i),
                f_reconstructed(&i),
                "Mismatch for function {} at input {}",
                name,
                i
            );
        }
        println!("All tests passed for function: {}\n", name);
    }

    #[test]
    fn is_constant_zero() {
        fn is_constant_zero(_: &usize) -> bool {
            return false;
        }

        test_isomorphism("is_constant_zero", &is_constant_zero);
    }

    #[test]
    fn is_constant_one() {
        fn is_constant_one(_: &usize) -> bool {
            return true;
        }

        test_isomorphism("is_constant_one", &is_constant_one);
    }

    #[test]
    fn is_divisible_by_3() {
        fn is_divisible_by_3(x: &usize) -> bool {
            return x % 3 == 0;
        }

        test_isomorphism("is_divisible_by_3", &is_divisible_by_3);
    }

    #[test]
    fn is_prime() {
        fn is_prime(x: &usize) -> bool {
            for i in 2..=(*x as f64).sqrt() as usize {
                if x % i == 0 {
                    return false;
                }
            }
            return true;
        }

        test_isomorphism("is_prime", &is_prime);
    }

    #[test]
    fn is_greater_than_42() {
        fn is_greater_than_42(x: &usize) -> bool {
            return *x > 42;
        }

        test_isomorphism("is_greater_than_42", &is_greater_than_42);
    }
}
