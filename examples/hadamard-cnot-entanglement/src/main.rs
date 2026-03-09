use quasim::{
    circuit::Circuit,
    simulator::{BuildSimulator, RunnableSimulator},
    sv_simulator::SVSimulator,
};
extern crate pretty_env_logger;

fn main() {
    pretty_env_logger::init();
    let circuit = Circuit::new(2).h(0).cnot(&[0], 1);
    let sim = match SVSimulator::build(circuit) {
        Ok(sim) => sim,
        Err(e) => panic!("Error building simulator: {}", e),
    };

    println!("\nResult: {:#04b}", sim.run());
}
