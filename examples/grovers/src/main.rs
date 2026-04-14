use std::env;

use quasim::circuit::{Circuit, HybridCircuit};
use quasim::debug_simulator::DebugSimulator;
use quasim::debug_terminal::DebugTerminal;
use quasim::sampler::RegisterSampler;
use quasim::simulator::{Buildable, Sampleable};
use quasim::sv_simulator::StateVectorSimulator;

fn check_quantum(func: &[usize]) -> bool {
    let fun_res: usize = func.iter().rev().enumerate().map(|(i, &b)| b << i).sum();

    StateVectorSimulator::sample_once(circuit(func), RegisterSampler::new("res")).unwrap()
        == fun_res
}

fn circuit(func: &[usize]) -> Circuit<HybridCircuit> {
    let bits: usize = func.len();

    let n = 1 << bits;
    let mut circuit = Circuit::new(bits)
        .new_reg("res", bits)
        .new_sub_circuit("u_f", create_oracle(func))
        .new_sub_circuit("g", create_diffusion(func));

    for i in 0..bits {
        circuit = circuit.h(i);
    }

    let iterations = (std::f64::consts::PI / 4.0 * ((n as f64).sqrt())).floor() as usize;

    for _i in 0..iterations {
        // Oracle
        circuit = circuit.call("u_f", 0);

        // Diffusion
        circuit = circuit.call("g", 0);
    }

    circuit = circuit.measure("res");
    circuit
}

fn create_oracle(func: &[usize]) -> Circuit {
    let bits = func.len();
    // Controlbits
    let c_array = &(0..(bits - 1)).collect::<Vec<_>>();
    let mut circuit = Circuit::new(func.len());

    for (j, &bit) in func.iter().rev().enumerate() {
        if bit == 0 {
            circuit = circuit.x(j);
        }
    }

    circuit = circuit.cz(&c_array, bits - 1);

    for (j, &bit) in func.iter().rev().enumerate() {
        if bit == 0 {
            circuit = circuit.x(j);
        }
    }

    circuit
}

fn create_diffusion(func: &[usize]) -> Circuit {
    let bits = func.len();
    // Controlbits
    let c_array = &(0..(bits - 1)).collect::<Vec<_>>();
    let mut circuit = Circuit::new(func.len());

    for i in 0..bits {
        circuit = circuit.h(i);
        circuit = circuit.x(i);
    }

    circuit = circuit.cz(&c_array, bits - 1);

    for i in 0..bits {
        circuit = circuit.x(i);
        circuit = circuit.h(i);
    }

    circuit
}

fn main() {
    for arg in env::args().skip(1) {
        if arg == "debug" {
            debug_main();
            return;
        }
    }

    let func: &[usize] = &[1, 0, 0]; // f(x) written as b_x,b_(x-1),...,b_0

    let iter = 1000;
    let mut true_count = 0;

    for _ in 0..iter {
        if check_quantum(func) {
            true_count += 1;
        }
    }

    println!("True count: {}", true_count);
}

fn debug_main() {
    let func: &[usize] = &[1, 0, 0]; // f(x) written as b_x,b_(x-1),...,b_0
    let circ = circuit(func);
    let sim: DebugSimulator = DebugSimulator::build(circ).expect("Could not build simulator");
    let mut term = DebugTerminal::from_simulator(sim);
    term.run().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grover() {
        let func: &[usize] = &[1, 0, 1]; // f(x) written as b_x,b_(x-1),...,b_0
        let iter = 1000;
        let mut true_count = 0;
        for _ in 0..iter {
            if check_quantum(func) {
                true_count += 1;
            }
        }

        assert!(true_count >= 1 - 1 / 2_i32.pow(func.len() as u32));
    }
}
