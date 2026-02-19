use quasim::{self, SimpleSimpleSimulator, SimpleSimulator};

fn main() {
    // let circuit = quasim::Circuit::new(2).hadamard(0).cnot(0, 1);
    let circuit = match quasim::Circuit::from_qasm_file("src/circuit.qasm") {
        Ok(circuit) => circuit,
        Err(e) => panic!("Error reading QASM file: {}", e),
    };
    let sim = match SimpleSimpleSimulator::build(circuit) {
        Ok(sim) => sim,
        Err(e) => panic!("Error building simulator: {}", e),
    };

    println!("\nResult: {:#04b}", sim.run());
}
