use quasim::circuit::{Circuit, HybridCircuit};
use quasim::simulator::{
    BuildSimulator, DebuggableSimulator, HybridSimulator, StoredCircuitSimulator,
};
use quasim::sv_simulator::SVSimulatorDebugger;

#[derive(PartialEq, Debug)]
pub enum FunctionType {
    Constant0,
    Constant1,
    Balanced,
}

pub fn find_function_type_quantum<S>(n: usize, function_type: FunctionType) -> bool
where
    S: BuildSimulator<HybridCircuit>
        + DebuggableSimulator
        + StoredCircuitSimulator
        + HybridSimulator,
{
    let mut circuit = Circuit::new(n + 1).new_reg("res", n);
    circuit = circuit.x(n);

    for i in 0..=n {
        circuit = circuit.h(i);
    }

    // Simple oracle
    match function_type {
        FunctionType::Constant0 => {}
        FunctionType::Constant1 => {
            circuit = circuit.x(n);
        }
        FunctionType::Balanced => {
            circuit = circuit.cx(&[0], n);
        }
    }

    for i in 0..n {
        circuit = circuit.h(i);
    }

    circuit = circuit.measure_bits(0..n, "res");
    let mut sim = S::build(circuit).unwrap();
    sim.cont();

    sim.register("res").read() == 0
}
