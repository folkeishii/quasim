use quasim::{
    circuit::Circuit,
    simulator::{BuildSimulator, RunnableSimulator},
    syntax_simulator::SyntaxSimulator,
};
extern crate pretty_env_logger;

fn main() {
    pretty_env_logger::init();

    let circuit = Circuit::new(2).h(0).cx(&[0], 1);
    let sim = match SyntaxSimulator::build(circuit) {
        Ok(sim) => sim,
        Err(e) => panic!("Error building simulator: {}", e),
    };

    println!("\nResult: {:#04b}", sim.run());
}
