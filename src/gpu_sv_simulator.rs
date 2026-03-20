use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl::{CubeCount, CubeDim, Runtime, cube};
use nalgebra::{Complex, DVector};
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::circuit::HybridCircuit;
use crate::ext::get_gate2_data;
use crate::{
    cart,
    circuit::{Circuit, pc::CircuitPc},
    expr_dsl::{Expr, Value},
    gate::{Gate, QBits},
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::StoredCircuitSimulator,
};

#[derive(Debug, Clone)]
pub struct GPUSVExecutor {
    state_vector_bytes: Bytes,
    state_vector_len: usize,

    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile<Value>,
}

impl GPUSVExecutor {
    fn new(circuit: Circuit<HybridCircuit>) -> Self {
        let size = 1 << circuit.n_qubits();

        // Bytes init
        let mut init_state: Vec<f64> = vec![0.0; size * 2]; // two floats for each complex number
        init_state[0] = 1.0; // Sets first complex re = 1.0
        let init_bytes = Bytes::from_elems(init_state);

        let registers = RegisterFile::from(circuit.registers());
        Self {
            state_vector_bytes: init_bytes,
            state_vector_len: size,
            circuit: circuit,
            pc: Default::default(),
            registers: registers,
        }
    }

    /// Step forward one instruction in the circuit
    pub fn step(&mut self) -> Option<&[Complex<f64>]> {
        let Some(inst) = self.circuit.instruction(self.pc()) else {
            return None;
        };

        self.apply_instruction(&inst);

        Some(self.state_vector())
    }

    /// Run the entire circuit
    pub fn step_all(&mut self) -> &Self {
        while let Some(_) = self.step() {}
        self
    }

    /// Gets a collapsed result from the current state vector
    pub fn get_collapsed_state(&self) -> usize {
        let probs = self.state_vector().iter().map(|&c| c.norm_sqr());

        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }

    /// Get current state of the quantum system
    pub fn state_vector(&self) -> &[Complex<f64>] {
        complex_from_bytes(&self.state_vector_bytes)
    }

    pub fn state_vector_mut(&mut self) -> &mut [Complex<f64>] {
        complex_from_bytes_mut(&mut self.state_vector_bytes)
    }

    fn launch_batched_gate2<'a, R: Runtime>(&mut self, gates: &[&'a Gate]) {
        let client = R::client(&Default::default());

        let n_gates = gates.len();
        let n_amplitudes = self.state_vector_len;

        // Batch data
        let mut gate_data: Vec<Complex<f64>> = Vec::with_capacity(n_gates * 4); // 4 complex amps per gate
        let mut target_data: Vec<u32> = Vec::with_capacity(n_gates);
        let mut control_data: Vec<u32> = Vec::with_capacity(n_gates);

        for &gate in gates {
            // Only 2x2 gates are suppsed to be passed to this function, so this *should* always be Some
            if let Some(data) = get_gate2_data(gate) {
                gate_data.extend(data);
                target_data.push(gate.get_target_bits().get_bitstring() as u32);
                control_data.push(gate.get_control_bits().get_bitstring() as u32);
            }
        }

        let state_handle = client.create_from_slice(&self.state_vector_bytes);
        let gate_data_handle = client.create_from_slice(bytes_from_complex(&gate_data));
        let target_data_handle = client.create_from_slice(u32::as_bytes(&target_data));
        let control_data_handle = client.create_from_slice(u32::as_bytes(&control_data));

        unsafe {
            let _ = batched_gate2_kernel::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(n_amplitudes as u32),
                ArrayArg::from_raw_parts::<f64>(&state_handle, n_amplitudes * 2, 1),
                ArrayArg::from_raw_parts::<u32>(&target_data_handle, n_gates, 1),
                ArrayArg::from_raw_parts::<u32>(&control_data_handle, n_gates, 1),
                ArrayArg::from_raw_parts::<f64>(&gate_data_handle, n_gates * 8, 1),
            );
        }

        let bytes = client.read_one(state_handle);
        self.state_vector_bytes = bytes;
    }

    fn get_batches<'a>(&mut self) -> Vec<Vec<&'a Gate>> {
        let batches: Vec<Vec<&'a Gate>> = Vec::new();
        let batch_count = 0;

        while let Some(Instruction::Gate(gate)) = &self.circuit.instruction(self.pc()) {
            
            

            self.pc_mut().increment();
        };

        batches
    }

    fn gate(&mut self, gate: &Gate) {
        let gates = &[gate];

        self.launch_batched_gate2::<WgpuRuntime>(gates);

        self.pc_mut().increment();
    }

    fn measure_bit(&mut self, target: usize, reg: &str, bit_pos: usize) {
        let measurement = self.get_collapsed_state();
        let mask = 1 << target;
        let measured_bit = measurement & mask;
        let shifted_measurement = ((measurement >> target) & 1) << bit_pos;
        let register_bit_mask = 1 << bit_pos;

        if let Value::Int(val) = self.registers[reg] {
            let val_cleared = (val as usize) & !register_bit_mask;
            self.registers[reg] = Value::Int((val_cleared | shifted_measurement) as i32)
        } else {
            self.registers[reg] = Value::Int(shifted_measurement as i32)
        }

        // Go through state vector and remove amplitude for all states that do not align with measurement
        for (i, amp) in self.state_vector_mut().iter_mut().enumerate() {
            if (i & mask) != measured_bit {
                *amp = Complex::ZERO;
            }
        }

        // Renormalize state vector
        let norm = self.state_vector()
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>()
            .sqrt();
        self.state_vector_mut().iter_mut().for_each(|x| *x /= norm);

        self.pc_mut().increment();
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement = self.get_collapsed_state();

        self.registers[reg] = Value::Int(measurement as i32);

        // Collapse whole state vector
        self.state_vector_mut().fill(cart!(0.0));
        self.state_vector_mut()[measurement] = cart!(1.0);

        self.pc_mut().increment();
    }

    fn jump(&mut self, label_pc: usize) {
        self.pc_mut().jump(label_pc);
    }

    fn jump_if(&mut self, expr: &Expr, label_pc: usize) {
        match expr.eval(&self.registers) {
            Ok(Value::Bool(true)) => self.jump(label_pc),
            Ok(Value::Bool(false)) => self.pc_mut().increment(),
            Err(err) => panic!("{}", err),
            _ => panic!(
                "Expression was expected to evaluate to boolean type but got something else."
            ),
        }
    }

    fn assign(&mut self, expr: &Expr, reg: &str) {
        match expr.eval(&self.registers) {
            Ok(value) => self.registers[reg] = value,
            Err(err) => panic!("{}", err),
        }
        self.pc_mut().increment();
    }

    fn apply_instruction(&mut self, inst: &Instruction) {
        match inst {
            Instruction::Gate(gate) => self.gate(gate),
            Instruction::MeasureBit(qbit, (reg, bit_pos)) => self.measure_bit(*qbit, reg, *bit_pos),
            Instruction::MeasureAll(reg) => self.measure_all(reg),
            Instruction::Jump(pc) => self.jump(*pc),
            Instruction::JumpIf(expr, pc) => self.jump_if(expr, *pc),
            Instruction::Assign(expr, reg) => self.assign(expr, reg),
        }
    }

    fn pc(&self) -> &CircuitPc {
        &self.pc
    }

    fn pc_mut(&mut self) -> &mut CircuitPc {
        &mut self.pc
    }
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
struct ComplexF64 {
    re: f64,
    im: f64,
}

#[cube]
impl ComplexF64 {
    fn mul(self, rhs: Self) -> Self {
        ComplexF64 {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    fn add(self, rhs: Self) -> Self {
        ComplexF64 {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

#[cube(launch)]
fn batched_gate2_kernel(
    state_vector: &mut Array<f64>,
    target_data: &Array<u32>,
    control_data: &Array<u32>,
    gate_data: &Array<f64>,
) {
    for i in 0..target_data.len() {
        let target = target_data[i] as usize;
        let control = control_data[i] as usize;

        apply_unitary2(state_vector, target, control, gate_data, i * 8);

        sync_storage();
    }
}

#[cube(launch)]
fn single_gate2_kernel(
    state_vector: &mut Array<f64>,
    target: u32,
    control: u32,
    gate_data: &Array<f64>,
) {
    apply_unitary2(
        state_vector,
        target as usize,
        control as usize,
        gate_data,
        0,
    );
}

#[cube]
fn apply_unitary2(
    state_vector: &mut Array<f64>,
    target: usize,
    control: usize,
    gate_data: &Array<f64>,
    gate_base: usize,
) {
    // ABSOLUTE_POS is basis state
    let basis0: usize = ABSOLUTE_POS * 2;
    let basis1: usize = (ABSOLUTE_POS | target) * 2;

    let is_block_base: bool = (ABSOLUTE_POS & target) == 0;
    let controls_active: bool = (ABSOLUTE_POS & control) == control;

    if is_block_base && controls_active {
        let u00 = ComplexF64 {
            re: gate_data[gate_base],
            im: gate_data[gate_base + 1],
        };
        let u01 = ComplexF64 {
            re: gate_data[gate_base + 2],
            im: gate_data[gate_base + 3],
        };
        let u10 = ComplexF64 {
            re: gate_data[gate_base + 4],
            im: gate_data[gate_base + 5],
        };
        let u11 = ComplexF64 {
            re: gate_data[gate_base + 6],
            im: gate_data[gate_base + 7],
        };

        let amp0 = ComplexF64 {
            re: state_vector[basis0],
            im: state_vector[basis0 + 1],
        };
        let amp1 = ComplexF64 {
            re: state_vector[basis1],
            im: state_vector[basis1 + 1],
        };

        let new_amp0 = u00.mul(amp0).add(u01.mul(amp1));
        let new_amp1 = u10.mul(amp0).add(u11.mul(amp1));

        state_vector[basis0] = new_amp0.re;
        state_vector[basis0 + 1] = new_amp0.im;

        state_vector[basis1] = new_amp1.re;
        state_vector[basis1 + 1] = new_amp1.im;
    }
}

fn bytes_from_complex<T: CubeElement>(data: &[Complex<T>]) -> &[u8] {
    let flat: &[T] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, data.len() * 2) };

    T::as_bytes(flat)
}

fn complex_from_bytes<T: CubeElement>(bytes: &[u8]) -> &[Complex<T>] {
    let data = T::from_bytes(&bytes);

    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const Complex<T>, data.len() / 2) }
}

fn bytes_from_complex_mut<T: CubeElement>(data: &mut [Complex<T>]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * size_of::<Complex<T>>()) }
}

fn complex_from_bytes_mut<T: CubeElement>(bytes: &mut Bytes) -> &mut [Complex<T>] {
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut Complex<T>, bytes.len() / size_of::<Complex<T>>()) }
}

impl StoredCircuitSimulator for GPUSVExecutor {
    type B = HybridCircuit;

    fn circuit(&self) -> &Circuit<HybridCircuit> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<HybridCircuit> {
        &mut self.circuit
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SVGPUError {}

mod tests {
    use nalgebra::DVector;

    use crate::{circuit::Circuit, gpu_sv_simulator::GPUSVExecutor};

    #[test]
    fn test() {
        let circ = Circuit::new(4)
            .h(0)
            .cx(&[0], 1)
            .cx(&[0], 2)
            .cx(&[0], 3);

        let mut exec = GPUSVExecutor::new(circ.into());

        exec.step_all();
        println!("{}", DVector::from_row_slice(exec.state_vector()))
    }
}
