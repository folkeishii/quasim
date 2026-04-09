use std::env;

use grovers::check_quantum;
use quasim::debug_simulator::DebugSimulator;
use quasim::debug_terminal::DebugTerminal;
use quasim::simulator::BuildSimulator;
use quasim::sv_simulator::SVSimulatorDebugger;

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
        if check_quantum::<SVSimulatorDebugger>(func) {
            true_count += 1;
        }
    }

    println!("True count: {}", true_count);
}

fn debug_main() {
    let func: &[usize] = &[1, 0, 0]; // f(x) written as b_x,b_(x-1),...,b_0
    let circ = grovers::circuit(func);
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
            if check_quantum::<SVSimulatorDebugger>(func) {
                true_count += 1;
            }
        }

        assert!(true_count >= 1 - 1 / 2_i32.pow(func.len() as u32));
    }
}
