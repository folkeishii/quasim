use std::io;

use quasim::circuit::Circuit;
use quasim::debug_simulator::DebugSimulator;
use quasim::debug_terminal::DebugTerminal;

fn main() -> io::Result<()> {
    let circuit = Circuit::new(3).hadamard(0).cnot(0, 2).cnot(2, 1);
    let mut term = DebugTerminal::<DebugSimulator>::new(circuit).expect("Test could not build debug terminal");

    term.run()
}
