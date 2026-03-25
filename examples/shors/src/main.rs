use quasim::circuit::Circuit;
use std::f64::consts::PI;
use quasim::simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator};
use quasim::sv_simulator::SVSimulatorDebugger;
use gcd::Gcd;
use rand::RngExt;

// Expects a QFT'd register of n qubits
fn adder(a: usize, n: usize) -> Circuit {
    let n_bits = (n as f64).log2().ceil() as usize;
    let mut c = Circuit::new(n+1);
    // Bitwise representation of a in the computational basis
    let a_bit_array = (0..n_bits)
    .map(|i| a & (1 << (n_bits - 1 - i)) != 0)
    .rev()
    .collect::<Vec<bool>>();

    // Apply rz to the second register based on the bits of a
    for i in 0..n {
        for j in a_bit_array[i..].iter() {
            if *j {
                let theta = 2.0 * PI / (1 << (i+1)) as f64;
                c = c.rz(theta, i);
            }
        }
    }

    c
}

fn mod_adder (a: usize, n: usize, circuit: Circuit) -> Circuit {
    let n_bits = (n as f64).log2().ceil() as usize;
    let c_array = (0..n_bits - 1).collect::<Vec<usize>>();

    circuit.adder(a,n);
    circuit.adder(n,n).rev();

    circuit.qft(&c_array).rev();
    circuit.cx(last_adder_bit, last_bit);
    circuit.qft(&c_array);

    circuit.adder(n,n);
    circuit.adder(a,n).rev();

    circuit.qft(&c_array).rev();
    circuit.x(last_adder_bit);
    circuit.cx(last_adder_bit, last_bit);
    circuit.x(last_adder_bit);
    circuit.qft(&c_array);

    circuit.adder(a,n);
}

fn cmult (a: usize, n: usize, circuit: Circuit) -> Circuit {
    let n_bits = (n as f64).log2().ceil() as usize;
    let c_array = (0..n_bits - 1).collect::<Vec<usize>>();

    circuit.qft(bottom_n_register);

    for i in 0..n_bits{
        circuit = circuit.mod_adder(2.pow(i)*a, n);
    }

    circuit.qft(bottom_n_register).rev();
}

fn u_a (a: usize, n: usize, circuit: Circuit) -> Circuit {
    let n_bits = (n as f64).log2().ceil() as usize;
    let c_array = (0..n_bits - 1).collect::<Vec<usize>>();

    circuit.cmult(a, n);
    circuit.swap(top_n_register, bottom_n_register);
    circuit.cmult(a, n).rev();
}

fn quantum(n: usize, a: usize) -> usize {
    let n_bits: usize = (n as f64).log2().ceil() as usize;
    let mut final_bit_array: Vec<usize> = vec![Default::default();2*n_bits];

    let mut circuit = Circuit::new(2*n_bits+3).new_reg("res");

    circuit = circuit.h(0);
    circuit = circuit.u_a(a, n);
    circuit = circuit.h(0);
    circuit = circuit.measure_bit(0, "reg");
    final_bit_array[0] = circuit.register("res").into_int().unwrap() as usize;

    for i in 0..2*n_bits-1 {

        // Check previous bit and apply X if 1
        if final_bit_array[i] == 1 {
            circuit = circuit.x(0);
        }

        circuit = circuit.h(0);
        circuit = circuit.u_a(a.pow(2.pow(i+1)), n);

        // R gates based on previous bits
        for j in 1..i+2 {
            if final_bit_array[j-1] == 1 {
                let theta = 2.0 * PI / (1 << (j+1)) as f64;
                circuit = circuit.rz(theta, 0);
            }
        }

        // Measure the next bit and store it in the result register
        circuit = circuit.measure_bit(0, "reg");
        final_bit_array[i+1] = circuit.register("res").into_int().unwrap() as usize;
    }

    // Return the measured bits as a number
    final_bit_array.iter().rev().enumerate().map(|(i, &b)| b << i).sum()
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


fn main() {
    let a = 15;

    println!("Bits of {}: {:?}", a, quant_adder(a));
}