use quasim::{
    circuit::Circuit,
    simulator::{Buildable, QuantumState, Simulator},
    sv_simulator::SVSimulator,
};
extern crate pretty_env_logger;

fn main() {
    pretty_env_logger::init();
    let circuit = match Circuit::from_qasm_file("src/circuit.qasm") {
        Ok(circuit) => circuit,
        Err(e) => panic!("Error reading QASM file: {}", e),
    };
    let mut sim = match SVSimulator::build(circuit) {
        Ok(sim) => sim,
        Err(e) => panic!("Error building simulator: {}", e),
    };

    println!("\nResult: {:#04b}", sim.run().state().collapse());
}
