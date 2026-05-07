use quasim::circuit::{Circuit, HybridCircuit};

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum FunctionType {
    Constant0 = 0,
    Constant1,
    Balanced,
}
impl From<usize> for FunctionType {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::Constant0,
            1 => Self::Constant1,
            2 => Self::Balanced,
            _ => panic!("Cannot construct FunctionType from {}", value)
        }
    }
}

pub fn circuit(n: usize, function_type: FunctionType) -> Circuit<HybridCircuit> {
    let mut circuit = Circuit::new(n + 1).new_reg("res", n);
    circuit = circuit.x(n);

    for i in 0..=n {
        circuit = circuit.h(i);
    }

    // Simple oracle
    match function_type {
        FunctionType::Constant0 => {}
        FunctionType::Constant1 => {
            circuit = circuit.x(n);
        }
        FunctionType::Balanced => {
            circuit = circuit.cx(&[0], n);
        }
    }

    for i in 0..n {
        circuit = circuit.h(i);
    }

    circuit.measure_bits(&(0..n).collect::<Vec<_>>(), "res")
}
