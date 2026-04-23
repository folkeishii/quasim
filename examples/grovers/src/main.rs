use std::env;

use grovers::circuit;
use quasim::{
    circuit::HybridCircuit,
    debug_terminal::DebugTerminal,
    fmm_simulator::FullMatMulSimulator,
    sampler::RegisterSampler,
    simulator::{Buildable, Sampleable, StoredRegisters},
    sv_simulator::StateVectorSimulator,
};

pub fn check_quantum<S: Buildable<HybridCircuit> + Sampleable<HybridCircuit> + StoredRegisters>(
    func: &[usize],
) -> bool {
    let fun_res: usize = func.iter().rev().enumerate().map(|(i, &b)| b << i).sum();
    S::sample_once(circuit(func), RegisterSampler::new("res")).unwrap() == fun_res
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
        if check_quantum::<StateVectorSimulator>(func) {
            true_count += 1;
        }
    }

    println!("True count: {}", true_count);
}

fn debug_main() {
    let func: &[usize] = &[1, 0, 0]; // f(x) written as b_x,b_(x-1),...,b_0
    let circ = grovers::circuit(func);
    let sim = FullMatMulSimulator::build(circ).expect("Could not build simulator");
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
            if check_quantum::<StateVectorSimulator>(func) {
                true_count += 1;
            }
        }

        assert!(true_count >= 1 - 1 / 2_i32.pow(func.len() as u32));
    }
}
