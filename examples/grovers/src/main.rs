use quasim::circuit::Circuit;
use quasim::expr_dsl::Value;
use quasim::simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator};
use quasim::sv_simulator::SVSimulatorDebugger;

const PI: f64 = 3.14; // Used in calculating the number of iterations for Grover's algorithm

fn check_quantum(func: &[usize]) -> bool {
    let bits: usize = func.len();
    let n = 1 << bits;
    let mut circuit = Circuit::new(bits).new_reg("res");

    for i in 0..bits {
        circuit = circuit.h(i);
    }

    let iterations = (PI / 4.0 * ((n as f64).sqrt())).floor() as usize;

    for _i in 0..iterations {
        // Controlbits
        let c_array = (0..bits - 1).collect::<Vec<usize>>();

        // Oracle
        for (j, &bit) in func.iter().rev().enumerate() {
            if bit == 0 {
                circuit = circuit.x(j);
            }
        }

        circuit = circuit.cz(&c_array, bits - 1);

        for (j, &bit) in func.iter().rev().enumerate() {
            if bit == 0 {
                circuit = circuit.x(j);
            }
        }

        // Diffusion
        for i in 0..bits {
            circuit = circuit.h(i);
            circuit = circuit.x(i);
        }

        circuit = circuit.cz(&c_array, bits - 1);

        for i in 0..bits {
            circuit = circuit.x(i);
            circuit = circuit.h(i);
        }
    }

    circuit = circuit.measure(&(0..bits).collect::<Vec<usize>>(), "res");

    let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
    sim.cont();

    let fun_res: usize = func.iter().rev().enumerate().map(|(i, &b)| b << i).sum();

    match sim.register("res") {
        Value::Int(x) => {
            let res = x == fun_res as i32;
            res
        }
        Value::Float(_) => {
            panic!("Unexpected float register")
        }
        Value::Bool(_) => {
            panic!("Unexpected bool register")
        }
    }
}

fn main() {
    let func: &[usize] = &[1, 0, 1]; // f(x) written as b_x,b_(x-1),...,b_0

    let iter = 1000;
    let mut true_count = 0;

    for _ in 0..iter {
        if check_quantum(func) {
            true_count += 1;
        }
    }

    println!("True count: {}", true_count);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grover() {
        let func: &[usize] = &[1, 0, 1]; // f(x) written as b_x,b_(x-1),...,b_0
        let iter = 1000;
        let mut true_count = 0;
        for _ in 0..iter {
            if check_quantum(func) {
                true_count += 1;
            }
        }

        assert!(true_count >= 1 - 1 / 2_i32.pow(func.len() as u32));
    }
}
