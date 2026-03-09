use std::io;

use quasim::circuit::Circuit;
use quasim::debug_simulator::DebugSimulator;
use quasim::debug_terminal::DebugTerminal;
// use quasim::sv_simulator::SVSimulatorDebugger;

fn main() -> io::Result<()> {
<<<<<<< HEAD
    let circuit = Circuit::new(3).h(0).cx(&[0], 2).cx(&[2], 1);
    let mut term = DebugTerminal::<SVSimulatorDebugger>::new(circuit)
=======
    let circuit = Circuit::new(3).hadamard(0).cnot(0, 2).cnot(2, 1);
    let mut term = DebugTerminal::<DebugSimulator>::new(circuit)
>>>>>>> 641be4e (removed sub circuits)
        .expect("Test could not build debug terminal");

    term.run()
}
