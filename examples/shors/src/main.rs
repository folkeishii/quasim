use num_integer::gcd;
use quasim::circuit::HybridCircuit;
use quasim::sampler::RegisterSampler;
use quasim::simulator::{Sampleable, StoredRegisters};
use rand::RngExt;
use shors::{circuit, modpow};

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

/// Refines a candidate period `k` by finding the smallest divisor `d`
/// such that `a^d ≡ 1 (mod n)`.
///
/// # Arguments
/// * `a` - The base used in modular exponentiation
/// * `n` -  The modulo number
/// * `k` - The initial period
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
pub fn qpe<S>(n: usize, a: usize) -> Option<usize>
where
    S: Sampleable<HybridCircuit> + StoredRegisters,
{
    let n_bits: usize = ((n as f32) + 1.0).log2().ceil() as usize;

    //let r = StateVectorSimulator::sample_once(circuit, RegisterSampler::new("res")).unwrap();
    let r = S::sample_once(circuit(n, a), RegisterSampler::new("res")).unwrap();

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
pub fn shors<S>(n: usize, a: usize) -> Option<Vec<usize>>
where
    S: Sampleable<HybridCircuit> + StoredRegisters,
{
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
pub fn shors_random<S>(n: usize, start: usize, stop: usize) -> Option<Vec<usize>>
where
    S: Sampleable<HybridCircuit> + StoredRegisters,
{
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
        sv_simulator::StateVectorSimulator,
    };

    use crate::{qpe, shors, shors_random};

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
