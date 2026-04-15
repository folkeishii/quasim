use num_integer::{Integer, gcd};
use quasim::circuit::{Circuit, HybridCircuit};
use quasim::expr_dsl::expr_helpers::rb;
use quasim::sampler::RegisterSampler;
use quasim::simulator::{Sampleable, StoredRegisters};
use rand::RngExt;
use std::f32::consts::PI;

/// Computes the modular inverse of `a` mod `n`.
///
/// # Arguments
/// * `a` - Number to compute modular inverse of
/// * `n` -  Number Modulo
///
/// # Returns
/// * `isize`, representing the inverse of `a` mod `n`
pub fn mod_inv(a: isize, n: isize) -> isize {
    let egcd = a.extended_gcd(&n);

    let mut inv = egcd.x % n;
    if inv < 0 {
        inv += n;
    }

    inv
}

/// Computes the exponential power of a number modulo.
///
/// # Arguments
/// * `b` - The base of the exponential
/// * `ex` -  The power to raise the `base` to
/// * `n` - The modulo number
///
/// # Returns
/// * `usize`, representing (`b`^`ex`) mod `n`
pub fn modpow(mut b: usize, mut ex: usize, n: usize) -> usize {
    let mut result = 1;
    b %= n;
    while ex > 0 {
        if ex % 2 == 1 {
            result = (result * b) % n;
        }
        b = (b * b) % n;
        ex /= 2;
    }
    result
}

/// Performs continued fractions using `num` and `den`
///
/// ```text
/// num / den = a0 + 1/(a1 + 1/(a2 + ...))
/// ```
///
/// # Arguments
/// * `num` - The numerator
/// * `den` - The denominator
///
/// # Returns
/// * `Vec<usize>`, representing the continued fraction coefficients `[a0, a1, ...]`
///
/// # Example
/// ```text
/// 415 / 93 → [4, 2, 6, 7]
/// ```
fn continued_fraction(mut num: usize, mut den: usize) -> Vec<usize> {
    let mut cf = Vec::new();

    while den != 0 {
        cf.push(num / den);
        let r = num % den;
        num = den;
        den = r;
    }
    cf
}

/// Computes the convergents of a continued fraction expansion
///
/// ```text
/// num / den = a0 + 1/(a1 + 1/(a2 + ...))
/// ```
///
/// # Arguments
/// * `cf` - Continued fraction coefficients `[a0, a1, ...]`
///
/// # Returns
/// * `Vec<(usize, usize)>`, representing a vector of convergents `(numerator, denominator)`
///
/// # Example
/// CF: `[4, 2, 6, 7]`
/// Returns:
/// ```text
/// (4/1), (9/2), (58/13), (415/93)
/// ```
fn convergents(cf: &[usize]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();

    let (mut h1, mut h2) = (1, 0);
    let (mut k1, mut k2) = (0, 1);

    for &a in cf {
        let h = a * h1 + h2;
        let k = a * k1 + k2;

        result.push((h, k));

        h2 = h1;
        h1 = h;
        k2 = k1;
        k1 = k;
    }
    result
}

/// Refines a candidate period `r` by finding the smallest divisor `d`
/// such that `a^d ≡ 1 (mod n)`.
///
/// # Arguments
/// * `a` - The base used in modular exponentiation
/// * `n` -  The modulo number
/// * `init_r` - The initial period
/// * `t` - The number of bits of precision
///
/// # Returns
/// * `Some(usize)`, if a valid period is found
/// * `None`, if not
fn extract_period(a: usize, n: usize, k: usize, t: usize) -> Option<usize> {
    let num = k;
    let den = 1 << t;

    let cf = continued_fraction(num, den);
    let convs = convergents(&cf);

    for &(_, r) in &convs {
        if r == 0 || r > n {
            continue;
        }

        for m in 1..=n {
            let candidate = r * m;
            if candidate > n {
                break;
            }

            if modpow(a, candidate, n) != 1 {
                continue;
            }

            if candidate % 2 != 0 {
                continue;
            }

            if modpow(a, candidate / 2, n) == n - 1 {
                continue;
            }

            return Some(candidate);
        }
    }
    None
}

/// Constructs an adder gate that sums two numbers and stores the
/// result in `n_bits` qubits.
///
/// The gate is intended to operate on a circuit that has already been
/// transformed by a Quantum Fourier Transform (QFT). The input state
/// is therefore assumed to be in the QFT basis.
///
/// The gate performs the transformation:
///
/// `|ϕ(b)⟩ --> |ϕ(a+b)⟩`
///
/// # Arguments
/// * `a` - First number to add
/// * `b` - Second number to add
///
/// # Returns
/// * `PureCircuit` representing the adder operation
fn create_adder(a: usize, b: usize) -> Circuit {
    // Number of bits needed to represent n and one overflow bit
    let n_bits = 1 + ((a as f32) + 1.0).log2().ceil() as usize;

    // Bitwise representation of a
    let a_bit_array = (0..n_bits)
        .map(|i| b & (1 << i) != 0)
        .collect::<Vec<bool>>();

    let mut circuit = Circuit::new(n_bits);

    // Apply rz to the qubits based on the bits of a
    for i in 0..n_bits {
        for j in 0..=i {
            if a_bit_array[j] {
                let bitshift = 1 << (i - j + 1);
                let theta = 2.0 * PI / bitshift as f32;
                circuit = circuit.rz(theta, n_bits - 1 - i)
            }
        }
    }

    circuit
}

/// Constructs a modular adder gate that adds the value `a` to the circuit mod `n`.
///
/// The gate is intended to operate on a circuit that has already been
/// transformed by a Quantum Fourier Transform (QFT). The input state
/// is therefore assumed to be in the QFT basis.
///
/// The gate performs the transformation:
///
/// `|ϕ(b)⟩ --> |ϕ((a+b) mod n)⟩`
///
/// # Arguments
/// * `n` - Modulo number
/// * `a` - Number to add
///
/// # Returns
/// * `PureCircuit` representing the modular adder operation
fn create_mod_adder(n: usize, a: usize) -> Circuit {
    // n-bits to represent the number being added to
    let n_bits = ((n as f32) + 1.0).log2().ceil() as usize;

    // Circuit has n-bits for the number, aswell as an overflow bit and a control bit, in that order
    let mut circuit = Circuit::new(n_bits + 2)
        .new_sub_circuit("adder_a", create_adder(n, a))
        .new_sub_circuit("adder_n", create_adder(n, n))
        .new_sub_circuit("adder_n_inv", create_adder(n, n).inverse())
        .new_sub_circuit("adder_a_inverse", create_adder(n, a).inverse())
        .new_sub_circuit("qft", Circuit::new_qft(n_bits + 1))
        .new_sub_circuit("qft_inv", Circuit::new_qft(n_bits + 1).inverse());

    circuit = circuit.call("adder_a", 0);
    circuit = circuit.call("adder_n_inv", 0);

    circuit = circuit.call("qft_inv", 0);
    circuit = circuit.cx(&[n_bits], n_bits + 1);
    circuit = circuit.call("qft", 0);

    circuit = circuit.ccall("adder_n", 0, &[n_bits + 1]);
    circuit = circuit.call("adder_a_inverse", 0);

    circuit = circuit.call("qft_inv", 0);
    circuit = circuit.x(n_bits);
    circuit = circuit.cx(&[n_bits], n_bits + 1);
    circuit = circuit.x(n_bits);
    circuit = circuit.call("qft", 0);

    circuit = circuit.call("adder_a", 0);

    circuit
}

/// Constructs a controlled multiplier gate that multiplies `a`
/// with the value of the first set of `n_bits` qubits and adds it
/// to the value of the second set of `n_bits` qubits, everything mod `n`.
///
/// The gate performs the transformation:
///
/// `|x|b⟩ --> |x|(b+a*x) mod n⟩`
///
/// # Arguments
/// * `n` - Modulo number
/// * `a` - Number to multiply with
///
/// # Returns
/// * `PureCircuit` representing the controlled multiplier operation
fn create_cmult(n: usize, a: usize) -> Circuit {
    let n_bits = ((n as f32) + 1.0).log2().ceil() as usize;

    let mut circuit = Circuit::new(2 * n_bits + 2);

    circuit = circuit.call_new("qft", Circuit::new_qft(n_bits + 1), n_bits);

    for i in 0..n_bits {
        circuit = circuit.ccall_new(
            format!("mod_adder{}", i),
            create_mod_adder(n, (a * (1 << i)) % n),
            n_bits,
            &[i],
        );
    }

    circuit = circuit.call_new("qft-inv", Circuit::new_qft(n_bits + 1).inverse(), n_bits);

    circuit
}

/// Constructs a swap gate that swaps the first set of
/// `n_bits` qubits with the second set of `n_bits` qubits.
///
/// The gate performs the transformation:
///
/// `|x|b⟩ --> |b|x⟩`
///
/// # Arguments
/// * `n` - Number with size `n_bits`, determines what qubits to target
///
/// # Returns
/// * `PureCircuit` representing the swap operation
fn create_swap(n: usize) -> Circuit {
    let n_bits = ((n as f32) + 1.0).log2().ceil() as usize;

    let mut circuit = Circuit::new(2 * n_bits);

    for i in 0..n_bits {
        circuit = circuit.swap(i, i + n_bits);
    }

    circuit
}

/// Constructs a controlled unitary gate that multiplies `a`
/// with the first set of `n_bits` qubits mod `n`.
///
/// The gate performs the transformation:
///
/// `|x⟩ --> |(a*x) mod n⟩`
///
/// # Arguments
/// * `n` - Modulo number
/// * `a` - Number to multiply with
///
/// # Returns
/// * `PureCircuit` representing the controlled multiplier operation
fn create_u_a(n: usize, a: usize) -> Circuit {
    let n_bits = ((n as f32) + 1.0).log2().ceil() as usize;

    let mut circuit = Circuit::new(2 * n_bits + 2)
        .new_sub_circuit("cmult", create_cmult(n, a))
        .new_sub_circuit("swap", create_swap(n))
        .new_sub_circuit(
            "inv_cmult",
            create_cmult(n, mod_inv(a as isize, n as isize) as usize).inverse(),
        );

    circuit = circuit.call("cmult", 0);
    circuit = circuit.call("swap", 0);
    circuit = circuit.call("inv_cmult", 0);

    circuit
}

/// Runs the Quantum Phase Estimation (QPE) algorithm used for period finding.
///
/// The circuit estimates the phase associated with the unitary operator
/// defined by modular multiplication, which is used to extract the order of `a` modulo `n`.
///
/// It can be expressed that the circuit estimates the eigenphase of the unitary operator U defined by:
///
/// `U|x⟩ = |(a*x) mod n⟩`
///
/// **NOTE: This algorithm is probalistic and may therefore fail to return a valid period although
/// one may exist for the chosen `a`.**
///
/// # Arguments
/// * `n` - Modulo number
/// * `a` - Base used in modular exponentiation
///
/// # Returns
/// * `Some(usize)`, if a valid period was found
/// * `None`, if not
pub fn qpe<S>(n: usize, a: usize) -> Option<usize> where S: Sampleable<HybridCircuit> + StoredRegisters {
    let n_bits: usize = ((n as f32) + 1.0).log2().ceil() as usize;

    let mut circuit = Circuit::new(2 * n_bits + 3).new_reg("res", 2 * n_bits);

    circuit = circuit.h(0);
    circuit = circuit.ccall_new("u_a0", create_u_a(n, a), 1, &[0]);
    circuit = circuit.h(0);
    circuit = circuit.measure_bit(0, ("res", 0));

    for i in 0..(2 * n_bits - 1) {
        // Check previous bit and apply X if 1
        circuit = circuit.apply_if(rb("res", i).eq(1)).x(0);

        circuit = circuit.h(0);
        circuit = circuit.ccall_new(
            format!("u_a{}", i + 1),
            create_u_a(n, modpow(a, 1 << (i + 1), n)),
            1,
            &[0],
        );

        // R gates based on previous bits
        for j in 0..i {
            let theta = -2.0 * PI / (1 << (i - j + 1)) as f32;
            circuit = circuit.apply_if(rb("res", j).eq(1)).rz(theta, 0);
        }

        // Measure the next bit and store it in the result register
        circuit = circuit.measure_bit(0, ("res", i + 1));
    }

    //let r = StateVectorSimulator::sample_once(circuit, RegisterSampler::new("res")).unwrap();
    let r = S::sample_once(circuit, RegisterSampler::new("res")).unwrap();

    extract_period(a, n, r, 2 * n_bits)
}

/// Runs Shor's algorithm to attempt to factor `n`.
///
/// # Arguments
/// * `n` - Number to factor
/// * `a` - Randomly chosen coprime to `n` in range [2, n-1] used for modular exponentation
///
/// **NOTE: This algorithm is probalistic and may therefore not return valid factors although
/// they may exist for the chosen `a`.**
///
/// # Returns
/// * `Some(Vec![f1,f2]` if non-trivial factors are found
/// * `None` if the attempt fails (due to an invalid period or quantum failure)
pub fn shors<S>(n: usize, a: usize) -> Option<Vec<usize>> where S: Sampleable<HybridCircuit> + StoredRegisters {
    if n % 2 == 0 {
        return Some(vec![2, n / 2]);
    }

    let g = gcd(a, n);
    if g > 1 {
        return Some(vec![g, n / g]);
    }

    let Some(r) = qpe::<S>(n, a) else {
        return None;
    };

    let a_pow = modpow(a, r / 2, n);

    // 4. reject bad cases
    if a_pow == 1 || a_pow == n - 1 {
        return None;
    }

    let factor1 = gcd(a_pow - 1, n);
    let factor2 = n / factor1;

    if factor1 != 1 && factor1 != n {
        return Some(vec![factor1, factor2]);
    }
    None
}

/// Runs Shor's algorithm to attempt to factor `n` with a randomized base
///
/// # Arguments
/// * `n` - Number to factor
/// * `start` - Lower bound for `a` (inclusive, minimum 2)
/// * `stop` - Upper bound for `a` (inclusive, maximum n - 1)
///
/// **NOTE: This algorithm is probalistic and may therefore not return valid factors although
/// they may exist in the chosen range. The function runs the algorithm 20 times but may still
/// fail to find valid factors.**
///
/// # Returns
/// * `Some(Vec![f1,f2]` if non-trivial factors are found
/// * `None` if the attempt fails (due to an invalid period or quantum failure)
pub fn shors_random<S>(n: usize, start: usize, stop: usize) -> Option<Vec<usize>> where S: Sampleable<HybridCircuit> + StoredRegisters {
    assert!(n > 1, "n must be > 1");
    assert!(start >= 2, "start must be >= 2");
    assert!(stop < n, "stop must be < n");
    assert!(start <= stop, "invalid range: start > stop");

    let mut rng = rand::rng();

    for _ in 0..20 {
        let a = rng.random_range(start..=stop);

        if gcd(a, n) != 1 {
            continue;
        }

        if let Some(factors) = shors::<S>(n, a) {
            return Some(factors);
        }
    }
    None
}

fn main() {
    let a = 4;
    let n = 5;

    println!("a: {}, n: {}", a, n);
}

#[cfg(test)]
mod tests {
    use quasim::{
        circuit::Circuit, sampler::RegisterSampler, simulator::Sampleable, sv_simulator::StateVectorSimulator
    };

    use crate::{
        create_adder, create_cmult, create_mod_adder, create_swap, create_u_a, mod_inv, qpe,
        shors, shors_random,
    };

    #[test]
    fn test_adder() {
        let n = 14;

        let n_bits = ((n as f32) + 1.0).log2().ceil() as usize;

        for a in 0..=n {
            let mut c = Circuit::new(n_bits + 1).new_reg("res", n_bits + 1);

            c = c.call_new("qft", Circuit::new_qft(n_bits + 1), 0);

            c = c.call_new("mod-adder", create_adder(n, a), 0);

            c = c.call_new("qft-inv", Circuit::new_qft(n_bits + 1).inverse(), 0);

            c = c.measure("res");

            let res = StateVectorSimulator::sample_once(c, RegisterSampler::new("res")).unwrap();

            // Constrained to the size of the number n
            assert_eq!(res, a % (1 << n_bits));
        }
    }

    #[test]
    fn test_mod_adder() {
        let n = 14;

        let n_bits = ((n as f32) + 1.0).log2().ceil() as usize;

        for a in 0..=n {
            let mut c = Circuit::new(n_bits + 2).new_reg("res", n_bits + 2);
            c = c.call_new("qft", Circuit::new_qft(n_bits + 1), 0);

            c = c.call_new("mod-adder", create_mod_adder(n, a), 0);

            c = c.call_new("qft-inv", Circuit::new_qft(n_bits + 1).inverse(), 0);

            c = c.measure("res");

            let res = StateVectorSimulator::sample_once(c, RegisterSampler::new("res")).unwrap();

            assert_eq!(res, a % n);
        }
    }

    #[test]
    fn test_cmult() {
        let n = 13;
        let x = [0, 1, 1];

        let n_bits = ((n as f32) + 1.0).log2().ceil() as usize;

        let c_array = (n_bits..=2 * n_bits).collect::<Vec<usize>>();

        for a in 2..n {
            let mut c = Circuit::new(2 * n_bits + 2).new_reg("res", n_bits + 1);

            for i in 0..x.len() {
                if x[i] == 1 {
                    c = c.x(i);
                }
            }

            c = c.call_new("cmult", create_cmult(n, a), 0);

            c = c.measure_bits(&c_array, "res");

            let res = StateVectorSimulator::sample_once(c, RegisterSampler::new("res")).unwrap();

            let x_tot: usize = x.iter().enumerate().map(|(i, &b)| b << i).sum();

            assert_eq!(res, (a * x_tot) % n);
        }
    }

    #[test]
    fn test_swap() {
        let n = 13;
        let x = [1, 1, 1];

        let n_bits = ((n as f32) + 1.0).log2().ceil() as usize;

        let mut c = Circuit::new(2 * n_bits)
            .new_reg("top", n_bits)
            .new_reg("bott", n_bits);

        for i in 0..x.len() {
            if x[i] == 1 {
                c = c.x(i);
            }
        }

        c = c.call_new("swap", create_swap(n), 0);

        c = c.measure_bits(&(0..n_bits).collect::<Vec<usize>>(), "top");
        c = c.measure_bits(&(n_bits..2 * n_bits).collect::<Vec<usize>>(), "bott");

        let top = StateVectorSimulator::sample_once(c.clone(), RegisterSampler::new("top")).unwrap();
        let bott = StateVectorSimulator::sample_once(c, RegisterSampler::new("bott")).unwrap();

        let x_tot: usize = x.iter().enumerate().map(|(i, &b)| b << i).sum();

        assert_eq!(top, 0);
        assert_eq!(bott, x_tot);
    }

    #[test]
    fn test_cmult_inv() {
        let n = 13;
        let y = [0, 1, 0];

        let n_bits = ((n as f32) + 1.0).log2().ceil() as usize;

        let c_array = (n_bits..2 * n_bits).collect::<Vec<usize>>();

        for a in 2..n {
            let a_inv = mod_inv(a as isize, n as isize) as usize;

            let mut c = Circuit::new(2 * n_bits + 2).new_reg("res", n_bits);

            for i in 0..y.len() {
                if y[i] == 1 {
                    c = c.x(i);
                }
            }

            c = c.call_new("cmult_inv", create_cmult(n, a_inv).inverse(), 0);

            c = c.measure_bits(&c_array, "res");

            let res = StateVectorSimulator::sample_once(c, RegisterSampler::new("res")).unwrap();

            let y_tot: usize = y.iter().enumerate().map(|(i, &b)| b << i).sum();

            let x = (a_inv * y_tot) % n;
            assert_eq!(res, (n - x) % n);
        }
    }

    #[test]
    fn test_u_a() {
        let n = 13;
        let x = [1, 1, 0];

        let n_bits = ((n as f32) + 1.0).log2().ceil() as usize;

        for a in 2..n {
            let mut c = Circuit::new(2 * n_bits + 2)
                .new_reg("top", n_bits)
                .new_reg("bott", n_bits);

            for i in 0..x.len() {
                if x[i] == 1 {
                    c = c.x(i);
                }
            }

            c = c.call_new("u_a", create_u_a(n, a), 0);

            c = c.measure_bits(&(0..n_bits).collect::<Vec<usize>>(), "top");
            c = c.measure_bits(&(n_bits..2 * n_bits).collect::<Vec<usize>>(), "bott");

            let top = StateVectorSimulator::sample_once(c.clone(), RegisterSampler::new("top")).unwrap();
            let bott = StateVectorSimulator::sample_once(c, RegisterSampler::new("bott")).unwrap();

            let x_t: usize = x.iter().enumerate().map(|(i, &b)| b << i).sum();

            assert_eq!(top, (x_t * a) % n);
            assert_eq!(bott, 0);
        }
    }

    #[test]
    fn test_quantum() {
        let n = 15;
        let a = 2;
        let attempts = 10;

        let mut success = false;

        for _ in 1..=attempts {
            let Some(_) = qpe::<StateVectorSimulator>(n, a) else {
                continue;
            };

            success = true;
            break;
        }
        assert!(
            success,
            "No non-trivial period found for a = {} in {} attempts.",
            a, attempts
        );
    }

    #[test]
    fn test_shors() {
        let n = 15;
        let a = 2;
        let attempts = 10;

        let mut success = false;

        for _ in 0..attempts {
            if let Some(res) = shors::<StateVectorSimulator>(n, a) {
                let f1 = res[0];
                let f2 = res[1];

                assert_eq!(f1 * f2, n);
                assert!(f1 > 1 && f2 > 1);

                success = true;
                break;
            }
        }
        assert!(
            success,
            "Shor failed to find factors with a = {} after {} attempts",
            a, attempts
        );
    }

    #[test]
    fn test_random_shors() {
        let n = 15;
        let start: usize = 2;
        let stop: usize = 10;
        let attempts: usize = 10;

        let mut success = false;

        for _ in 0..attempts {
            if let Some(res) = shors_random::<StateVectorSimulator>(n, start, stop) {
                let f1 = res[0];
                let f2 = res[1];

                assert_eq!(f1 * f2, n);
                assert!(f1 > 1 && f2 > 1);

                success = true;
                break;
            }
        }
        assert!(
            success,
            "Shor failed to find factors after {} attempts",
            attempts
        );
    }
}
