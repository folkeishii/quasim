use quasim::circuit::Circuit;
use quasim::expr_dsl::Value;
use quasim::simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator};
use quasim::sv_simulator::SVSimulatorDebugger;

const N: usize = 8;

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
    (x >> (N - 1)) == 1
}

#[allow(dead_code)]
fn f_balanced_2(x: u8) -> bool {
    x % 2 == 0
}

/// Check if a function is constant or balanced
#[allow(dead_code)]
fn check_classic(f: fn(u8) -> bool) -> FunctionType {
    let first = f(0);
    for i in 1..=(1 << (N - 1)) {
        if f(i) != first {
            return FunctionType::Balanced;
        }
    }

    if first {
        FunctionType::Constant1
    } else {
        FunctionType::Constant0
    }
}

/// Check if a function is constant or balanced
///
/// Return true if constant, false if balanced
fn check_quantum(function_type: FunctionType) -> bool {
    let mut circuit = Circuit::new(N + 1).new_reg("res");
    circuit = circuit.x(N);

    for i in 0..=N {
        circuit = circuit.h(i);
    }

    // Simple oracle
    match function_type {
        FunctionType::Constant0 => {}
        FunctionType::Constant1 => {
            circuit = circuit.x(N);
        }
        FunctionType::Balanced => {
            circuit = circuit.cx(&[0], N);
        }
    }

    for i in 0..N {
        circuit = circuit.h(i);
    }

    circuit = circuit.measure_bits(&[0, 1, 2, 3, 4, 5, 6, 7], "res");
    let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
    sim.cont();

    match sim.register("res") {
        Value::Int(i) => i == 0,
        Value::Float(_) => {
            panic!("Unexpected float register")
        }
        Value::Bool(_) => {
            panic!("Unexpected bool register")
        }
    }
}

fn main() {
    println!("{}", check_quantum(FunctionType::Constant0));
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
