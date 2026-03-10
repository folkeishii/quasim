use quasim::circuit::Circuit;
use quasim::expr_dsl::Value;
use quasim::simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator};
use quasim::sv_simulator::SVSimulatorDebugger;

const N: usize = 5;

fn f(x: u8, secret: u8) -> u8 {
    let mut sum = 0u8;

    for i in 0..N {
        sum += ((secret >> i) & 1) * ((x >> i) & 1);
    }

    sum % 2
}

fn find_secret_string_classical(f: impl Fn(u8) -> u8) -> u8 {
    let mut res = 0u8;

    for i in 0..N {
        res |= (f(1 << i) & 1) << i
    }

    res
}

fn find_secret_string_quantum(secret: u8) -> u8 {
    let mut circuit = Circuit::new(N + 1).new_reg("res");
    circuit = circuit.x(N);

    for i in 0..=N {
        circuit = circuit.h(i);
    }

    for i in 0..N {
        if (secret >> (N - (i + 1)) & 1) == 1 {
            circuit = circuit.cx(&[i], N);
        }
    }

    for i in 0..N {
        circuit = circuit.h(i);
    }

    circuit = circuit.measure(&[0, 1, 2, 3, 4], "res");
    let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
    sim.continue_until(None);

    match sim.register("res") {
        Value::Int(i) => {
            // Output is reversed
            (i as u8).reverse_bits() >> (8 - N)
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
    for i in 0..32u8 {
        println!(
            "{:b} - {:b}",
            find_secret_string_classical(|c| f(c, i)),
            find_secret_string_quantum(i)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bernstein_vazirani() {
        for i in 0..32u8 {
            assert_eq!(
                find_secret_string_classical(|c| f(c, i)),
                find_secret_string_quantum(i),
            )
        }
    }
}
