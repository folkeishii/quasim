use bernstein_vazirani::{find_secret_string_classical, find_secret_string_quantum};
use quasim::sv_simulator::SVSimulatorDebugger;

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
            find_secret_string_quantum::<SVSimulatorDebugger>(N, i)
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
                    find_secret_string_quantum::<SVSimulatorDebugger>(n, i),
                )
            }
        }
    }
}
