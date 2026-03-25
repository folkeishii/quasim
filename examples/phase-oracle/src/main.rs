use quasim::{
    circuit::Circuit,
    simulator::{BuildSimulator, RunnableSimulator},
    sv_simulator::SVSimulator,
};

fn is_prime(n: &usize) -> bool {
    if *n <= 1 {
        return false;
    }
    for i in 2..=((*n as f64).sqrt() as usize) {
        if *n % i == 0 {
            return false;
        }
    }
    true
}

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

fn main() {
    const N: usize = 7; // Number of qubits to input into the oracle
    let input_qubits: Vec<usize> = (0..N).collect();

    println!(
        "Testing primality oracle for numbers 0 to {}:",
        (1 << N) - 1
    );
    for i in 0..(1 << N) {
        let circuit = circuit_in_start_state(N + 1, i)
            .h(N)
            .phase_oracle(input_qubits.clone(), N, is_prime)
            .h(N);
        let sim = match SVSimulator::build(circuit) {
            Ok(sim) => sim,
            Err(e) => panic!("Error building simulator: {}", e),
        };

        let result = sim.run();
        let flipped = result & (1 << N) != 0;
        println!("For i={}:  {:#010b}, Flipped: {}", i, result, flipped);
    }
}
