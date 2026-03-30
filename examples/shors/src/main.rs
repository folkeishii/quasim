use quasim::circuit::{Circuit, HybridCircuit};
use quasim::expr_dsl::Value;
use quasim::expr_dsl::expr_helpers::r;
use std::env;
use std::f64::consts::PI;
use quasim::simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator};
use quasim::sv_simulator::SVSimulatorDebugger;
use gcd::Gcd;
use rand::RngExt;

// Adds a to the second n bits of the register
fn create_adder(n: usize, a: usize) -> Circuit {

    // Number of bits needed to represent n and one overflow bit
    let n_bits = 1 + ((n as f64)+1.0).log2().ceil() as usize;

    // Bitwise representation of a
    let a_bit_array = (0..n_bits)
    .map(|i| a & (1 << i) != 0)
    .collect::<Vec<bool>>();

    let mut circuit = Circuit::new(n_bits);

    // Apply rz to the qubits based on the bits of a
    for i in 0..n_bits {
        for j in 0..=i {
            if a_bit_array[j]{
                let bitshift = 1 << (i-j+1);
                let theta = 2.0 * PI / bitshift as f64;
                circuit = circuit.rz(theta, n_bits-1-i)
            }
        }
    }

    circuit
}

// Adds a to the second n bits of the register mod n
fn create_mod_adder (n: usize, a: usize) -> Circuit {

    // n-bits to represent the number being added to
    let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

    // QFT n_bits and the overflow bit
    let c_array = (0..n_bits+1).collect::<Vec<usize>>();

    // Circuit has n-bits for the number, aswell as an overflow bit and a control bit, in that order
    let mut circuit = Circuit::new(n_bits+2)
        .new_sub_circuit("adder_a", create_adder(n, a))
        .new_sub_circuit("adder_n", create_adder(n, n))
        .new_sub_circuit("adder_n_inv", create_adder(n, n).inverse())
        .new_sub_circuit("adder_a_inverse", create_adder(n, a).inverse())
        .new_sub_circuit("qft_inv", Circuit::new(n_bits).qft(&c_array).inverse());

    circuit = circuit.call("adder_a", 0);
    circuit = circuit.call("adder_n_inv", 0);

    circuit = circuit.call("qft_inv", 0);
    circuit = circuit.cx(&[n_bits], n_bits+1);
    circuit = circuit.qft(&c_array);

    circuit = circuit.ccall("adder_n", 0, &[n_bits+1]);
    circuit = circuit.call("adder_a_inverse", 0);

    circuit = circuit.call("qft_inv", 0);
    circuit = circuit.x(n_bits);
    circuit = circuit.cx(&[n_bits], n_bits+1);
    circuit = circuit.x(n_bits);
    circuit = circuit.qft(&c_array);

    circuit = circuit.call("adder_a", 0);

    circuit
}


/*
    Test check, current implementation ignores inner controls
    and treats them as outer controls. Check if this works
*/
fn create_cmult (n: usize, a: usize) -> Circuit {
    let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

    // QFT the second n_bits and its overflow bit
    let c_array = (n_bits..2*n_bits+1).collect::<Vec<usize>>();

    // Bitwise representation of a
    let a_bit_array = (0..n_bits)
    .map(|i| (a & (1 << i) != 0) as usize)
    .collect::<Vec<usize>>();

    let mut circuit = Circuit::new(2*n_bits+2);

    circuit = circuit.qft(&c_array);

    for i in 0..n_bits {
        circuit = circuit.breakpoint();
        circuit = circuit.ccall_new("mod_adder{i}",
            create_mod_adder(n, a_bit_array[i] * (1 << i)), n_bits,&[i]);
    }

    circuit = circuit.breakpoint();
    
    circuit = circuit.call_new("qft-inv", Circuit::new(n_bits+1).qft(&(0..n_bits+1).collect::<Vec<usize>>()).inverse(), n_bits);

    circuit
}

/*
fn create_swap(a: usize, n: usize) -> Circuit {
    let n_bits = (n as f64).log2().ceil() as usize;
    let c_array = (0..n_bits - 1).collect::<Vec<usize>>();

    let mut circuit = Circuit::new(2*n_bits);

    for i in 0..n_bits {
        circuit = circuit.swap(i, i+n_bits);
    }

    circuit
}

fn create_u_a (a: usize, n: usize) -> Circuit {
    let n_bits = (n as f64).log2().ceil() as usize;

    let mut circuit = Circuit::new(2*n_bits+2)
        .new_sub_circuit("cmult", create_cmult(n, a))
        .new_sub_circuit("inv_cmult", create_cmult(n, 1/a).inverse())
        .new_sub_circuit("swap", create_swap(a, n));

    circuit = circuit.call("cmult", 0);
    circuit = circuit.call("swap", 0);
    circuit = circuit.call("inv_cmult", 0);

    circuit
}

fn create_circuit(n: usize, a: usize) -> Circuit<HybridCircuit> {
    let n_bits: usize = (n as f64).log2().ceil() as usize;
    let mut final_bit_array: Vec<usize> = vec![Default::default();2*n_bits];

    let mut circuit = Circuit::new(2*n_bits+3)
        .new_reg("res");

    circuit = circuit.h(0);
    circuit = circuit.call_new("u_a0", create_u_a(a, n), 1).ctrl(0);
    circuit = circuit.h(0);
    circuit = circuit.measure_bit(0, ("res",0));

    for i in 0..2*n_bits-1 {

        // Check previous bit and apply X if 1
        circuit = circuit.apply_if(r("res").eq(1)).x(0);

        circuit = circuit.h(0);
        circuit = circuit.call_new(format!("u_a{}", i+1), create_u_a(a.pow(2.pow(i+1)), n), 1).ctrl(0);

        // R gates based on previous bits
        // TODO: Change to register based checking
        for j in 1..i+2 {
            if final_bit_array[j-1] == 1 {
                let theta = 2.0 * PI / (1 << (j+1)) as f64;
                circuit = circuit.rz(theta, 0);
            }
        }

        // Measure the next bit and store it in the result register
        circuit = circuit.measure_bit(0, ("res",i+1));
    }

    circuit
}

fn quantum(n: usize, a: usize) -> usize {
    let mut sim = SVSimulatorDebugger::build(create_circuit(n, a)).unwrap();
    sim.cont();
    
    match sim.register("res") {
        Value::Int(res) => {
            res as usize
        }
        Value::Float(_) => {
            panic!("Unexpected float register")
        }
        Value::Bool(_) => {
            panic!("Unexpected bool register")
        }
    }
}

fn shors(n: usize, init_a: usize) -> Vec<usize> {
    let mut a = init_a;
    loop {
        if n % 2 == 0 {
            return vec![2, n / 2];
        }

        if a.gcd(n) > 1 {
            return vec![n, a];
        }

        let l = quantum(n, a);

        if l == 0 {
            continue;
        }

        let b = l/n.pow(2);

        // Fractions

        let r = 1;

        if r % 2 != 0{
            let mut rng = rand::rng();
            a = rng.random_range(2..n);
            continue;
        }

        let n1 = (b.pow(r/2) + 1).gcd(n);

        if 1 < n1 && n1 < n {
            return vec![n1, n / n1];
        } else {
            let mut rng = rand::rng();
            a = rng.random_range(2..n);
            continue;
        }
    }
}

fn shors_random(n: usize) -> Vec<usize> {
    let a = n/2; // Randomly chosen coprime to n
    return shors(n, a);
}
*/

fn main() {

    let a = 2;
    let n = 3;
    let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

    let c_array = (0..2*n_bits+1).collect::<Vec<usize>>();

    let mut c = Circuit::new(2*n_bits+2).x(0).new_reg("res");
    c=c.breakpoint();
    c=c.call_new("cmult", create_cmult(n, a), 0);
    c=c.breakpoint();
    c=c.measure_bits(&c_array,"res");
    let mut sim = SVSimulatorDebugger::build(c).unwrap();

    sim.cont();
    println!("Initial state: {}", sim.current_state());

    sim.cont();
    
    for i in 0..n_bits{
        println!("State before {}: {}", i, sim.current_state());
        sim.cont();
    }

    println!("Before reverse QFT: {}", sim.current_state());

    sim.cont();
    
    println!("After cmult: {}", sim.current_state());

    sim.cont();

    println!("After measurement: {}", sim.current_state());



    

    match sim.register("res") {
        Value::Int(res) => {
            println!("Result: {}", res);
        }
        Value::Float(_) => {
            panic!("Unexpected float register")
        }
        Value::Bool(_) => {
            panic!("Unexpected bool register")
        }
    }
}