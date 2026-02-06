use quasim;
fn main() {
    let circuit = quasim::Circuit::default()
        .hadamard(0)
        .cnot(0, 1)
        .measure(None);

    todo!()
}
