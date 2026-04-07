use quasim::circuit::Circuit;
use quasim::simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator};
use quasim::sv_simulator::SVSimulatorDebugger;

#[allow(dead_code)]
#[derive(PartialEq, Debug)]
enum FunctionType {
    Constant0,
    Constant1,
    Balanced,
}

#[allow(dead_code)]
fn f_constant(_: u8) -> bool {
    false
}

#[allow(dead_code)]
fn f_constant_2(_: u8) -> bool {
    true
}

#[allow(dead_code)]
fn f_balanced(x: u8) -> bool {
    x == 0
}

#[allow(dead_code)]
fn f_balanced_2(x: u8) -> bool {
    x == 1
}

#[allow(dead_code)]
fn check_classic(f: fn(u8) -> bool) -> FunctionType {
    let first = f(0);
    let second = f(1);

    if first == second {
        if first {
            FunctionType::Constant1
        } else {
            FunctionType::Constant0
        }
    } else {
        FunctionType::Balanced
    }
}

/// Check if a function is constant or balanced
///
/// Return true if constant, false if balanced
fn check_quantum(function_type: FunctionType) -> bool {
    let mut circuit = Circuit::new(2).new_reg("res", 1);
    circuit = circuit.x(1);
    circuit = circuit.h(0).h(1);

    // Simple oracle
    match function_type {
        FunctionType::Constant0 => {}
        FunctionType::Constant1 => {
            circuit = circuit.x(1);
        }
        FunctionType::Balanced => {
            circuit = circuit.cx(&[0], 1);
        }
    }

    circuit = circuit.h(0).h(1);

    circuit = circuit.measure_bits([0], "res");
    let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
    sim.cont();

    sim.register("res").read() == 0
}

fn main() {
    println!("{}", check_quantum(FunctionType::Balanced));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum() {
        assert!(check_quantum(FunctionType::Constant0));
        assert!(check_quantum(FunctionType::Constant1));
        assert!(!check_quantum(FunctionType::Balanced));
    }

    #[test]
    fn test_classic() {
        assert_eq!(check_classic(f_constant), FunctionType::Constant0);
        assert_eq!(check_classic(f_constant_2), FunctionType::Constant1);
        assert_eq!(check_classic(f_balanced), FunctionType::Balanced);
        assert_eq!(check_classic(f_balanced_2), FunctionType::Balanced);
    }
}
