use quasim::circuit::Circuit;

pub fn circuit(n: usize, secret: usize) -> Circuit {
    assert_eq!(secret & !(usize::MAX << n), secret);

    let mut circuit = Circuit::new(n + 1);
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

    circuit
}
