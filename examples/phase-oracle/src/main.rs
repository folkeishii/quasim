/// n is the number of classical bits.
fn fn_to_truth_table(f: &dyn Fn(&usize) -> bool, n: usize) -> Vec<bool> {
    let mut truth_table = Vec::with_capacity(1 << n);
    for i in 0..(1 << n) {
        truth_table.push(f(&i));
    }
    return truth_table;
}

/// In-place transformation of the truth table to get the ANF coefficients.
fn anf_coefs(truth_table: &mut [bool]) {
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
}

/// Converts the ANF coefficients back to a function that can be evaluated on inputs.
fn anf_coefs_to_fn(anf_coefs: &[bool]) -> Box<dyn Fn(&usize) -> bool> {
    let anf_coefs = anf_coefs.to_vec(); // Clone the coefficients to move into the closure

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

#[cfg(test)]
fn test_isomorphism(name: &str, f: &dyn Fn(&usize) -> bool) {
    const N: usize = 12; // Number of bits for the truth table

    let mut truth_table = fn_to_truth_table(&f, N);
    println!("Truth Table for {}: {:?}", name, truth_table);
    anf_coefs(&mut truth_table);
    println!("ANF Coefficients for {}: {:?}", name, truth_table);
    let f_reconstructed = anf_coefs_to_fn(&truth_table);
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
fn test_is_constant_zero() {
    fn is_constant_zero(_: &usize) -> bool {
        return false;
    }

    test_isomorphism("is_constant_zero", &is_constant_zero);
}

#[test]
fn test_is_constant_one() {
    fn is_constant_one(_: &usize) -> bool {
        return true;
    }

    test_isomorphism("is_constant_one", &is_constant_one);
}

#[test]
fn test_is_divisible_by_3() {
    fn is_divisible_by_3(x: &usize) -> bool {
        return x % 3 == 0;
    }

    test_isomorphism("is_divisible_by_3", &is_divisible_by_3);
}

#[test]
fn test_is_prime() {
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
fn test_is_greater_than_42() {
    fn is_greater_than_42(x: &usize) -> bool {
        return *x > 42;
    }

    test_isomorphism("is_greater_than_42", &is_greater_than_42);
}

fn main() {
    let divisible_by_3 = |x: &usize| x % 3 == 0;
    let mut truth_table = fn_to_truth_table(&divisible_by_3, 4); // Generate truth table for 4 bits (16 entries)
    println!("Truth Table: {:?}", truth_table);
    anf_coefs(&mut truth_table);
    println!("ANF Coefficients: {:?}", truth_table);
    let f = anf_coefs_to_fn(&truth_table);
    for i in 0..(1 << 4) {
        println!("{} -> {:?}", i, f(&i));
    }
}
