use quasim::{circuit::Circuit, debug_terminal::DebugTerminal, simulator::BuildSimulator, sv_simulator::{SVSimulator, SVSimulatorDebugger}};

fn main() {
    let sub_circuit = Circuit::new(2).cnot(0, 1);
    let circuit = Circuit::new(3).new_sub_circuit("sub", sub_circuit).hadamard(0).cnot(0, 1).cnot(1, 2)
        .call("sub", 0)
        .call("sub", 1)
        .call("sub", 0)
        .call("sub", 1);

    let svsim = SVSimulator::build(circuit).unwrap();
    let debugger = svsim.attach_debugger();
    let mut term = DebugTerminal::<SVSimulatorDebugger<'_>>::from_simulator(debugger);
    term.run().unwrap()

}
