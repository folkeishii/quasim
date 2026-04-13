use quasim::{
    circuit::Circuit, sampler::CircuitSampler, simulator::Sampleable,
    sv_simulator::StateVectorSimulator,
};
extern crate pretty_env_logger;

fn main() {
    pretty_env_logger::init();
    let circuit = match Circuit::from_qasm_file("src/circuit.qasm") {
        Ok(circuit) => circuit,
        Err(e) => panic!("Error reading QASM file: {}", e),
    };

    let result = StateVectorSimulator::sample_once(circuit, &CircuitSampler)
        .expect("error building simulator");

    println!("\nResult: {:#04b}", result);
}
