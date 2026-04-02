use quasim::circuit::{Circuit, HybridCircuit};
use quasim::expr_dsl::Value;
use quasim::expr_dsl::expr_helpers::r;
use std::f64::consts::PI;
use std::ops::AddAssign;
use quasim::simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator};
use quasim::sv_simulator::SVSimulatorDebugger;
use gcd::Gcd;
use rand::RngExt;
use num_integer::{Integer};

// Computes the modular inverse of a mod n
pub fn mod_inv(a: isize, n: isize) -> isize {
    let egcd = a.extended_gcd(&n);

    let mut inv = egcd.x % n;
    if inv < 0 {
        inv += n;
    }

    inv
}

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
fn create_mod_adder(n: usize, a: usize) -> Circuit {

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


// Multiplies a and the first n_bits (x), stores the value in the second n_bits
// Disallows values a = n
fn create_cmult(n: usize, a: usize) -> Circuit {
    let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

    // QFT the second n_bits and its overflow bit
    let c_array = (n_bits..2*n_bits+1).collect::<Vec<usize>>();

    let mut circuit = Circuit::new(2*n_bits+2);

    circuit = circuit.qft(&c_array);

    for i in 0..n_bits {
        circuit = circuit.ccall_new(format!("mod_adder{}", i),
            create_mod_adder(n, (a * (1 << i))%n), n_bits, &[i]);
    }
    
    circuit = circuit.call_new("qft-inv", Circuit::new(n_bits+1).qft(&(0..n_bits+1).collect::<Vec<usize>>()).inverse(), n_bits);

    circuit
}

fn create_swap(n: usize) -> Circuit {
    let n_bits = ((n as f64)+1.0).log2().ceil() as usize;
    
    let mut circuit = Circuit::new(2*n_bits);

    for i in 0..n_bits {
        circuit = circuit.swap(i, i+n_bits);
    }

    circuit
}


fn create_u_a (n: usize, a: usize) -> Circuit {
    let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

    let mut circuit = Circuit::new(2*n_bits+2)
        .new_sub_circuit("cmult", create_cmult(n, a))
        .new_sub_circuit("swap", create_swap(n))
        .new_sub_circuit("inv_cmult", create_cmult(n, mod_inv(a as isize, n as isize) as usize).inverse());

    circuit = circuit.call("cmult", 0);
    circuit = circuit.call("swap", 0);
    circuit = circuit.call("inv_cmult", 0);

    circuit
}

fn create_quantum(n: usize, a: usize) -> Circuit<HybridCircuit> {
    let n_bits: usize = ((n as f64)+1.0).log2().ceil() as usize;
    let mut final_bit_array: Vec<usize> = vec![Default::default();2*n_bits];

    let mut circuit = Circuit::new(2*n_bits+3).new_reg("res");

    circuit = circuit.h(0);
    circuit = circuit.ccall_new("u_a0", create_u_a(a, n), 1,&[0]);
    circuit = circuit.h(0);
    circuit = circuit.measure_bit(0, ("res",0));

    for i in 0..2*n_bits-1 {

        // Check previous bit and apply X if 1
        circuit = circuit.apply_if(r("res").eq(1)).x(0);

        circuit = circuit.h(0);
        circuit = circuit.ccall_new(format!("u_a{}", i+1), create_u_a(n,a.pow(1 << (i+1))), 1, &[0]);
        
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
/* 
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

    let a = 4;
    let n = 5;
    let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

    let c_array = (n_bits..=2*n_bits).collect::<Vec<usize>>();

    let mut c = Circuit::new(2*n_bits+2).x(1).new_reg("res");
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

#[cfg(test)]
mod tests{
    use quasim::{circuit::Circuit, expr_dsl::Value, simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator, RunnableSimulator}, sv_simulator::{SVSimulator, SVSimulatorDebugger}};

    use crate::{create_adder, create_cmult, create_mod_adder, create_swap, create_u_a, mod_inv};

    #[test]
    fn test_adder(){
        let n = 14;

        let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

        let c_array = (0..=n_bits).collect::<Vec<usize>>();

        for a in 0..=n{

            let mut c = Circuit::new(n_bits+1).qft(&c_array);
            
            c = c.call_new("mod-adder", create_adder(n, a), 0);

            c = c.call_new("qft-inv", Circuit::new(n_bits+1).qft(&c_array).inverse(), 0);

            
            let sim = SVSimulator::build(c).unwrap();
        
            // Constrained to the size of the number n
            assert_eq!(sim.run(), a % (1 << n_bits));
        }
    }
    

    #[test]
    fn test_mod_adder(){
        let n = 14;

        let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

        let c_array = (0..=n_bits).collect::<Vec<usize>>();

        for a in 0..=n{

            let mut c = Circuit::new(n_bits+2).qft(&c_array);
            
            c = c.call_new("mod-adder", create_mod_adder(n, a), 0);

            c = c.call_new("qft-inv", Circuit::new(n_bits+1).qft(&(0..n_bits+1).collect::<Vec<usize>>()).inverse(), 0);

            
            let sim = SVSimulator::build(c).unwrap();

        assert_eq!(sim.run(), a % n);
        }
    }

    #[test]
    fn test_cmult(){
        let n = 13;
        let x = [0,1,1];


        let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

        let c_array = (n_bits..=2*n_bits).collect::<Vec<usize>>();

        for a in 2..n{

            let mut c = Circuit::new(2*n_bits+2).new_reg("res");

            for i in 0..x.len(){
                if x[i] == 1{
                    c = c.x(i);
                }
            }
            
            c = c.call_new("cmult", create_cmult(n, a), 0);

            c = c.measure_bits(&c_array, "res");
            
            let mut sim = SVSimulatorDebugger::build(c).unwrap();

            sim.cont();

            let x_tot: usize = x.iter().enumerate().map(|(i, &b)| b << i).sum();

            match sim.register("res") {
                Value::Int(res) => {
                    assert_eq!(res as usize,(a*x_tot)%n);
                }
                Value::Float(_) => {
                    panic!("Unexpected float register")
                }
                Value::Bool(_) => {
                    panic!("Unexpected bool register")
                }
            }
        }
    }

    #[test]
    fn test_swap(){
        let n = 13;
        let x = [1,1,1];

        let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

        let mut c = Circuit::new(2*n_bits).new_reg("top").new_reg("bott");

        for i in 0..x.len(){
            if x[i] == 1{
                c = c.x(i);
            }
        }

        c = c.call_new("swap", create_swap(n), 0);

        c = c.measure_bits(&(0..n_bits).collect::<Vec<usize>>(), "top");
        c = c.measure_bits(&(n_bits..2*n_bits).collect::<Vec<usize>>(), "bott");

        let mut sim = SVSimulatorDebugger::build(c).unwrap();

        sim.cont();

        match sim.register("top") {
            Value::Int(top) => {
                assert_eq!(top as usize, 0);
            }
            Value::Float(_) => {
                panic!("Unexpected float register")
            }
            Value::Bool(_) => {
                panic!("Unexpected bool register")
            }
        }

        match sim.register("bott") {
            Value::Int(bott) => {
                assert_eq!(bott as usize, x.iter().enumerate().map(|(i, &b)| b << i).sum());
            }
            Value::Float(_) => {
                panic!("Unexpected float register")
            }
            Value::Bool(_) => {
                panic!("Unexpected bool register")
            }
        }
    }

    #[test]
    fn test_cmult_inv(){
        let n = 13;
        let y = [0,1,0];


        let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

        let c_array = (n_bits..2*n_bits).collect::<Vec<usize>>();

        for a in 2..n{
            let a_inv = mod_inv(a as isize, n as isize) as usize;

            let mut c = Circuit::new(2*n_bits+2).new_reg("res");

            for i in 0..y.len(){
                if y[i] == 1{
                    c = c.x(i);
                }
            }
            
            c = c.call_new("cmult_inv", create_cmult(n, a_inv).inverse(), 0);

            c = c.measure_bits(&c_array, "res");
            
            let mut sim = SVSimulatorDebugger::build(c).unwrap();

            sim.cont();

            let y_tot: usize = y.iter().enumerate().map(|(i, &b)| b << i).sum();

            match sim.register("res") {
                Value::Int(res) => {
                    let x = (a_inv * y_tot) % n;
                    assert_eq!(res as usize, (n - x) % n);
                }
                Value::Float(_) => {
                    panic!("Unexpected float register")
                }
                Value::Bool(_) => {
                    panic!("Unexpected bool register")
                }
            }
        }
    }

    #[test]
    fn test_u_a(){
        let n = 13;
        let x = [1,1,0];

        let n_bits = ((n as f64)+1.0).log2().ceil() as usize;

        for a in 2..n{

            let mut c = Circuit::new(2*n_bits+2).new_reg("top").new_reg("bott");

            for i in 0..x.len(){
                if x[i] == 1{
                    c = c.x(i);
                }
            }
            
            c = c.call_new("u_a", create_u_a(n, a), 0);

            c = c.measure_bits(&(0..n_bits).collect::<Vec<usize>>(), "top");
            c = c.measure_bits(&(n_bits..2*n_bits).collect::<Vec<usize>>(), "bott");
            
            let mut sim = SVSimulatorDebugger::build(c).unwrap();

            sim.cont();


            let x_t: usize = x.iter().enumerate().map(|(i, &b)| b << i).sum();

            match sim.register("top") {
                Value::Int(top) => {
                    assert_eq!(top as usize, (x_t*a)%n);
                }
                Value::Float(_) => {
                    panic!("Unexpected float register")
                }
                Value::Bool(_) => {
                    panic!("Unexpected bool register")
                }
            }

            match sim.register("bott") {
                Value::Int(bott) => {
                    assert_eq!(bott as usize, 0);
                }
                Value::Float(_) => {
                    panic!("Unexpected float register")
                }
                Value::Bool(_) => {
                    panic!("Unexpected bool register")
                }
            }
        }
    }

    #[test]
    fn test_quantum(){

        let n = 6;
        let x = [1,1,0];
        let a = 3;

        let n_bits = ((n as f64)+1.0).log2().ceil() as usize;
    }
}