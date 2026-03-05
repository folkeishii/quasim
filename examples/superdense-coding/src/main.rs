use quasim::circuit::Circuit;
use quasim::expr_dsl::expr_helpers::r;
use quasim::expr_dsl::{Expr, Value};
use quasim::simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator};
use quasim::sv_simulator::SVSimulatorDebugger;

/// Simulate sending a 2-bit integer using one qubit
///
/// Input is the 2-bit integer to send and returns
/// the received 2-bit integer
fn send_int(i: u8) -> u8 {
    let c = i >> 1 & 1;
    let d = i & 1;

    #[rustfmt::skip]
    let circuit = Circuit::new(2)
        .new_reg("a0")
        .new_reg("a1")
        .new_reg("b")

        // Bell state is produced, alice gets q0 and bob q1
        .hadamard(0)
        .cnot(0, 1)

        // Alice has two bits, c and d
        .assign("a0".to_string(), Expr::Val(Value::Int(c as i32)))
        .assign("a1".to_string(), Expr::Val(Value::Int(d as i32)))

        .apply_if(r("a1").eq(1)).z(0)
        .apply_if(r("a0").eq(1)).x(0)

        // Alice sends qubit to bob, bob then performs cnot and hadamard
        .cnot(0, 1)
        .hadamard(0)

        // Bob measures his qubit + the received one to get c and d
        .measure(&[0, 1], "b");

    let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
    sim.continue_until(None);

    return match sim.register("b") {
        Value::Int(s) => s as u8,
        Value::Float(_) => {
            panic!("Unexpected float register")
        }
        Value::Bool(_) => {
            panic!("Unexpected bool register")
        }
    };
}

fn main() {
    for i in 0..4 {
        println!("{} - {}", i, send_int(i));
    }
}

#[cfg(test)]
mod tests {
    use crate::send_int;

    #[test]
    fn test_superdense_coding() {
        for i in 0..4 {
            assert_eq!(i, send_int(i))
        }
    }
}
