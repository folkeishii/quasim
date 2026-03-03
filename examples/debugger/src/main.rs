use std::io;

use quasim::circuit::Circuit;
//use quasim::debug_simulator::DebugSimulator;
use quasim::debug_terminal::DebugTerminal;
use quasim::sv_simulator::SVSimulatorDebugger;

fn main() -> io::Result<()> {
    let circuit = Circuit::new(3)
        .hadamard(0)
        .cnot(0, 2)
        .cnot(2, 1)
        .measure_bit(0, "a0")
        .measure_bit(1, "a1")
        .measure_bit(2, "a2");
    let mut term =
        DebugTerminal::<DebugSimulator>::new(circuit).expect("Test could not build debug terminal");

    term.run()
}
