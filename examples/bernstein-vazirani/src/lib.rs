use quasim::{
    circuit::{Circuit, HybridCircuit},
};

pub fn circuit(n: usize, secret: usize) -> Circuit<HybridCircuit> {
    assert_eq!(secret & !(usize::MAX << n), secret);

    let mut circuit = Circuit::new(n + 1).new_reg("res", n);
    circuit = circuit.x(n);

    for i in 0..=n {
        circuit = circuit.h(i);
    }

    for i in 0..n {
        if (secret >> (n - (i + 1)) & 1) == 1 {
            circuit = circuit.cx(&[i], n);
        }
    }

    for i in 0..n {
        circuit = circuit.h(i);
    }

    circuit = circuit.measure_bits(&(0..n).collect::<Vec<_>>(), "res");

    circuit
}
