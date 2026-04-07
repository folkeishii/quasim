use quasim::{
    circuit::{Circuit, HybridCircuit},
    simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator, StoredCircuitSimulator},
};

pub fn find_secret_string_classical(n: usize, f: impl Fn(usize) -> usize) -> usize {
    let mut res = 0;

    for i in 0..n {
        res |= (f(1 << i) & 1) << i
    }

    res
}

fn circuit(n: usize, secret: usize) -> Circuit<HybridCircuit> {
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

    circuit = circuit.measure_bits(0..n, "res");

    circuit
}

pub fn find_secret_string_quantum<S>(n: usize, secret: usize) -> usize
where
    S: BuildSimulator<HybridCircuit>
        + DebuggableSimulator
        + StoredCircuitSimulator
        + HybridSimulator,
{
    let mut sim = S::build(circuit(n, secret)).unwrap();
    sim.cont();

    let res = sim.register("res").read();
    // Output is reversed
    (res).reverse_bits() >> (size_of::<usize>() * 8 - n)
}
