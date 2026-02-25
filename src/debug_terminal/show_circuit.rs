use std::{
    fmt::Display,
    io::{self, Write},
};

use crate::simulator::DebuggableSimulator;

const LANE_MARGIN: usize = 1;
const GATE_PADDING: Size = Size {
    width: 1,
    height: 1,
};

struct Pos {
    lane: usize,
    column: usize,
}
struct Size {
    width: usize,
    height: usize,
}

enum Marker {
    Ket(usize),
    Gate { name: String, size: Size },
    SkipGate { width: usize },
    Measure,
}

fn ket<W: Write, T: Display>(write: &mut W, value: &T) -> io::Result<()> {
    print!(write; "|{}⟩", value)
}

// fn print_circuit<W: Write, S: DebuggableSimulator + >(write: &mut W, simulator)
