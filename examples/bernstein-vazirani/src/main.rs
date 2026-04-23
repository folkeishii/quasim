use bernstein_vazirani::circuit;
use quasim::{
    circuit::HybridCircuit,
    sampler::RegisterSampler,
    simulator::{Buildable, Sampleable, StoredRegisters},
    sv_simulator::StateVectorSimulator,
};

pub fn find_secret_string_classical(n: usize, f: impl Fn(usize) -> usize) -> usize {
    let mut res = 0;

    for i in 0..n {
        res |= (f(1 << i) & 1) << i
    }

    res
}

pub fn find_secret_string_quantum<S>(n: usize, secret: usize) -> usize
where
    S: Buildable<HybridCircuit> + Sampleable<HybridCircuit> + StoredRegisters,
{
    let res = S::sample_once(circuit(n, secret), RegisterSampler::new("res")).unwrap();
    // Output is reversed
    (res).reverse_bits() >> (size_of::<usize>() * 8 - n)
}

fn f(n: usize, x: usize, secret: usize) -> usize {
    let mut sum = 0;

    for i in 0..n {
        sum += ((secret >> i) & 1) * ((x >> i) & 1);
    }

    sum % 2
}

fn main() {
    const N: usize = 5;
    for i in 0..(1 << N) {
        println!(
            "{:b} - {:b}",
            find_secret_string_classical(N, |c| f(N, c, i)),
            find_secret_string_quantum::<StateVectorSimulator>(N, i)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bernstein_vazirani() {
        for n in 1..5 {
            for i in 0..(1 << n) {
                assert_eq!(
                    find_secret_string_classical(n, |c| f(n, c, i)),
                    find_secret_string_quantum::<StateVectorSimulator>(n, i),
                )
            }
        }
    }
}
