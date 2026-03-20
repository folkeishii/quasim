use std::ops::{Add, Mul};

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::{CubeCount, CubeDim, Runtime, cube};
use nalgebra::{Complex, DVector, DVectorView, Matrix2};
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::circuit::{CircuitBehaviour, HybridCircuit};
use crate::ext::{get_gate2_data, get_u_matrix2};
use crate::gate::GateType;
use crate::{
    cart,
    circuit::{Circuit, pc::CircuitPc},
    expr_dsl::{Expr, Value},
    gate::{Gate, QBits},
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{DebuggableSimulator, RunnableSimulator, StoredCircuitSimulator},
};

#[derive(Debug, Clone)]
pub struct GPUSVExecutor {
    state_vector: DVector<Complex<f64>>,
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    registers: RegisterFile<Value>,
}

impl GPUSVExecutor {
    fn new(circuit: Circuit<HybridCircuit>) -> Self {
        let size = 1 << circuit.n_qubits();
        let mut init_state_vector: DVector<Complex<f64>> = DVector::from_element(size, cart![0.0]);
        init_state_vector[0] = cart![1.0];

        let registers = RegisterFile::from(circuit.registers());
        Self {
            state_vector: init_state_vector,
            circuit: circuit,
            pc: Default::default(),
            registers: registers,
        }
    }

    /// Step forward one instruction in the circuit
    pub fn step(&mut self) -> Option<&DVector<Complex<f64>>> {
        let Some(inst) = self.circuit.instruction(self.pc()) else {
            return None;
        };

        self.apply_instruction(&inst);

        Some(&self.state_vector)
    }

    /// Run the entire circuit
    pub fn step_all(&mut self) -> &Self {
        while let Some(_) = self.step() {}
        self
    }

    /// Gets a collapsed result from the current state vector
    pub fn get_collapsed_state(&self) -> usize {
        let probs = self.state_vector.iter().map(|&c| c.norm_sqr());

        let dist = WeightedIndex::new(probs)
            .expect("Failed to create probability distribution. Invalid or empty state vector?");
        let mut rng = rand::rng();

        dist.sample(&mut rng)
    }

    /// Get current state of the quantum system
    pub fn state_vector(&self) -> &DVector<Complex<f64>> {
        &self.state_vector
    }

    /// Checks that all control bits are 1
    fn controls_active(i: usize, controls: QBits) -> bool {
        let control_mask = controls.get_bitstring();
        (i & control_mask) == control_mask
    }

    /// Checks that all target bits are 0
    fn is_block_base(i: usize, targets: QBits) -> bool {
        let target_mask = targets.get_bitstring();
        (i & target_mask) == 0
    }

    // 0 1
    // 1 0
    #[inline(always)]
    fn apply_x(&mut self, base_index: usize, target: QBits) {
        self.state_vector
            .as_mut_slice()
            .swap(base_index, base_index | target.get_bitstring());
    }

    // 0 -i
    // i  0
    #[inline(always)]
    fn apply_y(&mut self, base_index: usize, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let state = self.state_vector.as_mut_slice();
        let a = state[base_index];
        let b = state[flipped_index];

        state[base_index] = cart!(b.im, -b.re);
        state[flipped_index] = cart!(-a.im, a.re);
    }

    // 1  0
    // 0 -1
    #[inline(always)]
    fn apply_z(&mut self, base_index: usize, target: QBits) {
        let i = base_index | target.get_bitstring();
        let amp = &mut self.state_vector[i];

        amp.re = -amp.re;
        amp.im = -amp.im;
    }

    #[inline(always)]
    fn apply_h(&mut self, base_index: usize, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let state = self.state_vector.as_mut_slice();
        let a = state[base_index];
        let b = state[flipped_index];
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;

        state[base_index] = (a + b) * inv_sqrt2;
        state[flipped_index] = (a - b) * inv_sqrt2;
    }

    // #[inline(always)]
    // fn apply_rx(&mut self, base_index: usize, target: QBits) {}

    // #[inline(always)]
    // fn apply_ry(&mut self, base_index: usize, target: QBits) {}

    // #[inline(always)]
    // fn apply_rz(&mut self, base_index: usize, target: QBits) {}

    // #[inline(always)]
    // fn apply_phase(&mut self, base_index: usize, target: QBits) {}

    fn apply_s(&mut self, base_index: usize, target: QBits) {
        let i = base_index | target.get_bitstring();
        let amp = self.state_vector[i];

        self.state_vector[i].re = -amp.im;
        self.state_vector[i].im = amp.re;
    }

    #[inline(always)]
    fn apply_swap(&mut self, base_index: usize, targets: QBits) {
        let t0 = targets.get_indices()[0];
        let t1 = targets.get_indices()[1];

        let i01 = base_index | (1 << t0);
        let i10 = base_index | (1 << t1);

        self.state_vector.as_mut_slice().swap(i01, i10);
    }

    #[inline(always)]
    fn apply_unitary2(&mut self, base_index: usize, u: &Matrix2<Complex<f64>>, target: QBits) {
        let flipped_index = base_index | target.get_bitstring();
        let a = self.state_vector[base_index];
        let b = self.state_vector[flipped_index];

        self.state_vector[base_index] = u[(0, 0)] * a + u[(0, 1)] * b;
        self.state_vector[flipped_index] = u[(1, 0)] * a + u[(1, 1)] * b;
    }

    fn gate(&mut self, gate: &Gate) {
        let controls = gate.get_control_bits();
        let targets = gate.get_target_bits();
        let n = self.state_vector.len();

        // No parallelization
        // State vector is length 2^n , n=num qubits
        for i in 0..n {
            if !Self::is_block_base(i, targets) {
                continue;
            }

            if !Self::controls_active(i, controls) {
                continue;
            }

            match gate.get_type() {
                GateType::X => self.apply_x(i, targets),
                GateType::Y => self.apply_y(i, targets),
                GateType::Z => self.apply_z(i, targets),
                GateType::H => self.apply_h(i, targets),
                GateType::S => self.apply_s(i, targets),
                GateType::SWAP => self.apply_swap(i, targets),
                GateType::U(theta, phi, lambda) => {
                    self.apply_unitary2(i, &get_u_matrix2(theta, phi, lambda), targets)
                }
            }
        }

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
        for (i, amp) in self.state_vector.iter_mut().enumerate() {
            if (i & mask) != measured_bit {
                *amp = Complex::ZERO;
            }
        }

        // Renormalize state vector
        let norm = self
            .state_vector
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>()
            .sqrt();
        self.state_vector.iter_mut().for_each(|x| *x /= norm);

        self.pc_mut().increment();
    }

    fn measure_all(&mut self, reg: &str) {
        let measurement = self.get_collapsed_state();

        self.registers[reg] = Value::Int(measurement as i32);

        // Collapse whole state vector
        self.state_vector.fill(cart!(0.0));
        self.state_vector[measurement] = cart!(1.0);

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
    target_data: &Array<usize>,
    control_data: &Array<usize>,
    gate_data: &Array<f64>,
) {
    for i in 0..target_data.len() {
        let target = target_data[i];
        let control = control_data[i];
        
        apply_gate2(state_vector, target, control, gate_data, i * 8);
        
        sync_storage();
    }
}

#[cube]
fn apply_gate2(
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

fn launch_batched_gate2<'a, R: Runtime>(gates: &[&'a Gate]) {
    let client = R::client(&Default::default());

    let dv = DVector::from_element(128, cart![0.0]);

    let num_gates = gates.len();

    // Batch data
    let mut gate_data: Vec<Complex<f64>> = Vec::with_capacity(num_gates * 4); // 4 complex amps per gate
    let mut target_data: Vec<u32> = Vec::with_capacity(num_gates);
    let mut control_data: Vec<u32> = Vec::with_capacity(num_gates);

    for &gate in gates {
        // Only 2x2 gates are suppsed to be passed to this function, so this *should* always be Some
        if let Some(data) = get_gate2_data(gate) {
            gate_data.extend(data);
            target_data.push(gate.get_target_bits().get_bitstring() as u32);
            control_data.push(gate.get_control_bits().get_bitstring() as u32);
        }
    }

    let state_handle = client.create_from_slice(bytes_from_complex(dv.as_slice()));
    let gate_data_handle = client.create_from_slice(bytes_from_complex(&gate_data));
    let target_data_handle = client.create_from_slice(u32::as_bytes(&target_data));
    let control_data_handle = client.create_from_slice(u32::as_bytes(&control_data));

    unsafe {
        let _ = batched_gate2_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(128),
            ArrayArg::from_raw_parts::<f64>(&state_handle, dv.len(), 1),
            ArrayArg::from_raw_parts::<f64>(&gate_data_handle, num_gates * 8, 1),
            ArrayArg::from_raw_parts::<u32>(&target_data_handle, num_gates, 1),
            ArrayArg::from_raw_parts::<u32>(&control_data_handle, num_gates, 1),
        );
    }

    let bytes = client.read_one(state_handle);
    let result = complex_from_bytes::<f64>(&bytes);

    println!("{:?}", result);
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
    use cubecl::{
        Runtime,
        wgpu::{WebGpu, WgpuRuntime},
    };

    use crate::gpu_sv_simulator::launch_batched_gate2;

    #[test]
    fn test() {
        // launch_batched_gate2::<WgpuRuntime>(&Default::default());
    }
}
