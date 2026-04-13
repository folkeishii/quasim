use quasim::{
    circuit::Circuit, sampler::CircuitSampler, simulator::Sampleable,
    sv_simulator::StateVectorSimulator,
};
extern crate pretty_env_logger;

fn main() {
    pretty_env_logger::init();
    let circuit = Circuit::new(2).h(0).cx(&[0], 1);

    let result = StateVectorSimulator::sample_once(circuit, &CircuitSampler).unwrap();
    println!("\nResult: {:#04b}", result);
}
