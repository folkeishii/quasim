extern crate quasim;

use quasim::{
    circuit::Circuit,
    simulator::{BuildSimulator, RunnableSimulator},
    syntax_simulator::SyntaxSimulator,
};

fn main() {
    divan::main();
}

#[divan::bench(args = [8, 12, 16], sample_count = 8)]
fn controlled_u_with_superposition_controls(n_controlled_u_gates: usize) {
    let controls = [0, 1, 2, 3, 4];
    let target = 5;

    let mut circuit = Circuit::new(6);
    for c in controls {
        circuit = circuit.h(c);
    }

    for i in 0..n_controlled_u_gates {
        let x = i as f64;
        // Deterministic angles for stable benchmark inputs.
        let theta = (x * 0.017).sin();
        let phi = (x * 0.031).cos();
        let lambda = (x * 0.013).sin() * (x * 0.007).cos();
        circuit = circuit.cu(theta, phi, lambda, &controls, target);
    }

    let sim = SyntaxSimulator::build(circuit).expect("Couldnt build circuit...");
    let _ = sim.final_state();
}
