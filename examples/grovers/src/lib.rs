use quasim::{
    circuit::{Circuit, HybridCircuit},
    simulator::{BuildSimulator, DebuggableSimulator, HybridSimulator, StoredCircuitSimulator},
};

pub fn check_quantum<
    S: BuildSimulator<HybridCircuit> + DebuggableSimulator + StoredCircuitSimulator + HybridSimulator,
>(
    func: &[usize],
) -> bool {
    let mut sim = S::build(circuit(func)).unwrap();
    sim.cont();

    let fun_res: usize = func.iter().rev().enumerate().map(|(i, &b)| b << i).sum();

    sim.register("res").read() == fun_res
}

pub fn circuit(func: &[usize]) -> Circuit<HybridCircuit> {
    let bits: usize = func.len();

    let n = 1 << bits;
    let mut circuit = Circuit::new(bits)
        .new_reg("res", bits)
        .new_sub_circuit("u_f", create_oracle(func))
        .new_sub_circuit("g", create_diffusion(func));

    for i in 0..bits {
        circuit = circuit.h(i);
    }

    let iterations = (std::f64::consts::PI / 4.0 * ((n as f64).sqrt())).floor() as usize;

    for _i in 0..iterations {
        // Oracle
        circuit = circuit.call("u_f", 0);

        // Diffusion
        circuit = circuit.call("g", 0);
    }

    circuit = circuit.measure("res");
    circuit
}

fn create_oracle(func: &[usize]) -> Circuit {
    let bits = func.len();
    // Controlbits
    let c_array = &(0..(bits - 1)).collect::<Vec<_>>();
    let mut circuit = Circuit::new(func.len());

    for (j, &bit) in func.iter().rev().enumerate() {
        if bit == 0 {
            circuit = circuit.x(j);
        }
    }

    circuit = circuit.cz(&c_array, bits - 1);

    for (j, &bit) in func.iter().rev().enumerate() {
        if bit == 0 {
            circuit = circuit.x(j);
        }
    }

    circuit
}

fn create_diffusion(func: &[usize]) -> Circuit {
    let bits = func.len();
    // Controlbits
    let c_array = &(0..(bits - 1)).collect::<Vec<_>>();
    let mut circuit = Circuit::new(func.len());

    for i in 0..bits {
        circuit = circuit.h(i);
        circuit = circuit.x(i);
    }

    circuit = circuit.cz(&c_array, bits - 1);

    for i in 0..bits {
        circuit = circuit.x(i);
        circuit = circuit.h(i);
    }

    circuit
}
