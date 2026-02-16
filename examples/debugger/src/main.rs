use std::{io, thread::sleep, time::Duration};

use quasim::{Circuit, DebugTerminal};

fn main() -> io::Result<()> {
    let circuit = Circuit::new(3).hadamard(0).cnot(0, 2).cnot(2, 1);
    let mut term = DebugTerminal::new(circuit).expect("Test could not build debug terminal");

    term.run()
}
