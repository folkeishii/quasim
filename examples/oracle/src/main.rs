use quasim::{
    circuit::Circuit,
    simulator::{Buildable, QuantumState, Simulator},
    sv_simulator::SVSimulator,
};

/// Sets up a quantum circuit with the specified state of the qubits at the start.
fn circuit_in_start_state(n: usize, mut starting_qubits_mask: usize) -> Circuit {
    let mut circuit = Circuit::new(n);
    let mut i = 0;
    while starting_qubits_mask != 0 {
        if starting_qubits_mask & 1 == 1 {
            circuit = circuit.x(i);
        }
        starting_qubits_mask >>= 1;
        i += 1;
    }
    circuit
}

/// Runs a quantum oracle for a given input and function,
/// returning the collapsed final state of the qubits.
fn run_one_quantum_oracle(input: usize, f: impl Fn(usize) -> bool) -> usize {
    let input_qubits: Vec<usize> = (0..N).collect();
    let circuit = circuit_in_start_state(N + 1, input).oracle(&input_qubits, N, f);
    let mut sim = match SVSimulator::build(circuit) {
        Ok(sim) => sim,
        Err(e) => panic!("Error building simulator: {}", e),
    };

    sim.run().state().collapse()
}

/// Tests a quantum oracle against a classical function by comparing the results.
/// Tries all possible inputs for the specified number of qubits.
/// panics if the oracle does not match the function for any input, otherwise prints the results.
fn oracle_proof(f: &impl Fn(usize) -> bool) {
    for i in 0..(1 << N) {
        let result = run_one_quantum_oracle(i, f);
        let flipped = result & (1 << N) != 0;
        assert_eq!(
            flipped,
            f(i),
            "f({}) -> {}, but oracle did{} invert.",
            i,
            f(i),
            if !flipped { " not" } else { "" }
        );
        println!("For i={}:  {:#09b}, Flipped: {}", i, result, flipped);
    }
}

fn is_even(n: usize) -> bool {
    n % 2 == 0
}

fn is_divisible_by_7(n: usize) -> bool {
    n % 7 == 0
}

fn is_between_10_and_20(n: usize) -> bool {
    n >= 10 && n <= 20
}

fn is_prime(n: usize) -> bool {
    if n <= 1 {
        return false;
    }
    for i in 2..=((n as f64).sqrt() as usize) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

const N: usize = 6; // Number of qubits to input into the oracle

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn quantum_is_even() {
        oracle_proof(&is_even);
    }

    #[test]
    fn quantum_is_divisible_by_7() {
        oracle_proof(&is_divisible_by_7);
    }

    #[test]
    fn quantum_is_between_10_and_20() {
        oracle_proof(&is_between_10_and_20);
    }

    #[test]
    fn quantum_is_prime() {
        oracle_proof(&is_prime);
    }
}

fn main() {
    println!(
        "Testing for even numbers for numbers 0 to {}:",
        (1 << N) - 1
    );
    oracle_proof(&is_even);

    println!(
        "Testing for divisibility by 7 for numbers 0 to {}:",
        (1 << N) - 1
    );
    oracle_proof(&is_divisible_by_7);

    println!(
        "Testing for numbers between 10 and 20 for numbers 0 to {}:",
        (1 << N) - 1
    );
    oracle_proof(&is_between_10_and_20);

    println!(
        "Testing primality oracle for numbers 0 to {}:",
        (1 << N) - 1
    );
    oracle_proof(&is_prime);
}
