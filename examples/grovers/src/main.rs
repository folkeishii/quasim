use quasim::circuit::Circuit;
use quasim::expr_dsl::Value; 
use quasim::simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator};
use quasim::sv_simulator::SVSimulatorDebugger;

const BITS: usize = 2; // Number of input bits
const PI: f64 = 3.14; // Used in calculating the number of iterations for Grover's algorithm

const FUNC: [usize; BITS] = [0, 1]; // f(x) written as b_x,b_(x-1),...,b_0

fn check_quantum(function: [usize; BITS]) -> bool {
    let n = 1 << BITS-1;
    let mut circuit = Circuit::new(BITS).new_reg("res");

    for i in 0..BITS {
        circuit = circuit.hadamard(i);
    }

    for _i in 0..(PI/4.0*((n as f64).sqrt())) as usize {
        
        // Oracle
        for (j, &bit) in function.iter().enumerate() {
            if bit == 1{
                circuit = circuit.z(j)
            }
        }

        // Diffusion
        for i in 0..BITS-1 {
            circuit = circuit.hadamard(i);
        }
        circuit = circuit.z(BITS-1);

        circuit = circuit.cnot(0, 1);

        for i in 0..BITS-1 {
            circuit = circuit.hadamard(i);
        }
        circuit = circuit.z(BITS-1);
    }

   
    circuit = circuit.measure(&(0..BITS).collect::<Vec<usize>>(), "res");

    let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
    sim.continue_until(None);
    println!("State: {}",sim.current_state());

    let fun_res: usize = FUNC.iter().rev().enumerate().map(|(i, &b)| b << i).sum();

    match sim.register("res"){
        Value::Int(x) => {
            let res = x == fun_res as i32;
            if !res {
                panic!("expected {}, got {}", fun_res, x);
            }
            res
        },
        Value::Float(_) => {
            panic!("Unexpected float register")
        }
        Value::Bool(_) => {
            panic!("Unexpected bool register")
        }
    } 
}

fn main(){
    println!("{}", check_quantum(FUNC));
}