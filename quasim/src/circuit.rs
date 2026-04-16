use std::{collections::HashMap, f32::consts::PI};
pub mod breakpoint;
pub mod oracle;
pub mod pc;

use crate::{
    circuit::{
        breakpoint::{Breakpoint, BreakpointList, IEBreakpoint},
        oracle::{append_oracle, fn_to_truth_table, truth_table_to_anf_coefs},
        pc::CircuitPc,
    },
    expr_dsl::{BitExpr, BoolExpr},
    gate::{Gate, GateType, QBits},
    instruction::{Instruction, PureInstruction},
};
mod qasm_parse;

use log::{trace, warn};
use oq3_syntax::{SourceFile, ast::AstNode};
use std::fs::read_to_string;

pub use qasm_parse::*;

#[derive(Debug, Clone)]
pub struct Circuit<B: CircuitBehaviour = PureCircuit> {
    instructions: Vec<B::InstructionTy>,
    n_qubits: usize,
    labels: HashMap<String, usize>,
    unresolved_labels: Vec<(String, usize)>,
    breakpoints: BreakpointList,
    registers: HashMap<String, usize>,
    sub_circuits: HashMap<String, Circuit>,
}

// Pure specific
impl Circuit {
    pub fn new(n_qubits: usize) -> Circuit<PureCircuit> {
        Self {
            instructions: Vec::<PureInstruction>::default(),
            n_qubits: n_qubits,
            registers: HashMap::new(),
            labels: HashMap::new(),
            unresolved_labels: Vec::new(),
            breakpoints: Default::default(),
            sub_circuits: Default::default(),
        }
    }

    /// Creates a new circuit implementing the qft algorithm
    pub fn new_qft(n_qubits: usize) -> Self {
        let s = Self {
            instructions: Vec::<PureInstruction>::default(),
            n_qubits: n_qubits,
            registers: HashMap::new(),
            labels: HashMap::new(),
            unresolved_labels: Vec::new(),
            breakpoints: Default::default(),
            sub_circuits: Default::default(),
        };

        s.qft(&(0..n_qubits).collect::<Vec<_>>())
    }

    /// Creates a new circuit implementing a quantum oracle for a given classical function.
    /// `input_qubits` are specified in order of least significance, for example [0,1,2,3,4].
    pub fn new_oracle(
        n_qubits: usize,
        input_qubits: &[usize],
        target: usize,
        classic_fn: impl Fn(usize) -> bool,
    ) -> Self {
        let s = Self::new(n_qubits);

        s.oracle(input_qubits, target, classic_fn)
    }

    pub fn from_qasm_file(file_name: &str) -> Result<Self, QASMParseError> {
        let file_string = read_to_string(file_name)?;
        let parsed_source = SourceFile::parse(&file_string);
        let parse_tree: SourceFile = parsed_source.tree();
        trace!(
            "Found {} QASM statements",
            parse_tree.statements().collect::<Vec<_>>().len()
        );
        let syntax_errors = parsed_source.errors();
        if syntax_errors.len() > 0 {
            warn!(
                "Found {} QASM parse errors:\n{:?}\n",
                syntax_errors.len(),
                syntax_errors
            );
        }

        // First pass: count the number of qubits
        let n_qubits = count_qubits_from_syntax_tree(parse_tree.syntax())?;

        // Second pass: build the circuit by applying gates
        let circuit = apply_gates_from_syntax_tree(Circuit::new(n_qubits), parse_tree.syntax())?;

        return Ok(circuit);
    }

    /// Inverts a non-hybrid circuit.
    pub fn inverse(&self) -> Self {
        let mut inverted_circuit = Circuit::new(self.n_qubits());
        for instruction in self.instructions.iter().rev() {
            match instruction {
                PureInstruction::Gate(gate) => {
                    inverted_circuit.instructions.push(gate.inverse().into())
                }
                PureInstruction::Call(name, lsq, ctrl) => inverted_circuit
                    .instructions
                    .push(PureInstruction::Call(name.clone(), *lsq, *ctrl)),
            }
        }

        // Invert sub circuits
        for (name, circuit) in self.sub_circuits.iter() {
            inverted_circuit
                .sub_circuits
                .insert(name.clone(), circuit.inverse());
        }

        inverted_circuit
    }

    pub fn instruction(&self, circuit_pc: &CircuitPc) -> Option<PureInstruction> {
        if let Some((name, sub_pc)) = circuit_pc.next_sub_pc() {
            self.sub_circuits[name].instruction(sub_pc)
        } else {
            match self.instructions().get(circuit_pc.pc()) {
                Some(PureInstruction::Gate(gate)) => {
                    let mut gate = gate.clone() << circuit_pc.lsq();
                    *gate.control_mut() |= circuit_pc.ctrl();
                    Some(gate.into())
                }
                rst => rst.cloned(),
            }
        }
    }

    /// ## Returns
    /// Returns the circuit pointed at by `circuit_pc`
    pub fn current_circuit(&self, circuit_pc: &CircuitPc) -> &Circuit {
        if let Some((name, pc)) = circuit_pc.next_sub_pc() {
            self.sub_circuits[name].current_circuit(pc)
        } else {
            self
        }
    }

    /// ## Returns
    /// Returns an iterator that iterates over all instructions and hides the call instructions
    pub fn as_flat(&self) -> FlatCircuit<'_, PureCircuit> {
        FlatCircuit {
            circuit: self,
            pc: CircuitPc::new(0),
        }
    }
}

// Hybrid specific
impl Circuit<HybridCircuit> {
    pub fn instruction(&self, circuit_pc: &CircuitPc) -> Option<Instruction> {
        if let Some((name, sub_pc)) = circuit_pc.next_sub_pc() {
            self.sub_circuits[name]
                .instruction(sub_pc)
                .map(HybridCircuit::from_pure)
        } else {
            match self.instructions().get(circuit_pc.pc()) {
                Some(Instruction::Gate(gate)) => {
                    let mut gate = gate.clone() << circuit_pc.lsq();
                    *gate.control_mut() |= circuit_pc.ctrl();
                    Some(Instruction::Gate(gate))
                }
                Some(Instruction::MeasureBit(target, register)) => Some(Instruction::MeasureBit(
                    *target << circuit_pc.lsq(),
                    register.clone(),
                )),
                rst => rst.cloned(),
            }
        }
    }

    /// ## Returns
    /// Returns an iterator that iterates over all instructions and hides the call instructions
    pub fn as_flat(&self) -> FlatCircuit<'_, HybridCircuit> {
        FlatCircuit {
            circuit: self,
            pc: CircuitPc::new(0),
        }
    }
}

impl<B: CircuitBehaviour> Circuit<B> {
    pub fn valid_pc(&self, circuit_pc: &CircuitPc) -> bool {
        circuit_pc.pc() <= self.instructions().len()
    }

    pub fn instructions(&self) -> &[B::InstructionTy] {
        &self.instructions
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn registers(&self) -> &HashMap<String, usize> {
        &self.registers
    }

    pub fn has_unresolved_labels(&self) -> bool {
        !self.unresolved_labels.is_empty()
    }

    // Builder methods

    pub fn x(mut self, target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::X, &[], &[target]).unwrap().into(),
        ));
        self
    }

    pub fn cx(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::X, controls, &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn y(mut self, target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::Y, &[], &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn cy(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::Y, controls, &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn z(mut self, target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::Z, &[], &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn cz(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::Z, controls, &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn h(mut self, target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::H, &[], &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn ch(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::H, controls, &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn swap(mut self, target1: usize, target2: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::SWAP, &[], &[target1, target2])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn cswap(mut self, controls: &[usize], target1: usize, target2: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::SWAP, controls, &[target1, target2])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn u(mut self, theta: f32, phi: f32, lambda: f32, target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::U(theta, phi, lambda), &[], &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn cu(
        mut self,
        theta: f32,
        phi: f32,
        lambda: f32,
        controls: &[usize],
        target: usize,
    ) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::U(theta, phi, lambda), controls, &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn s(mut self, target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::S, &[], &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn cs(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::S, controls, &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn rx(mut self, theta: f32, target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::U(theta, -PI / 2.0, PI / 2.0), &[], &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn crx(mut self, theta: f32, controls: &[usize], target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::U(theta, -PI / 2.0, PI / 2.0), controls, &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn ry(mut self, theta: f32, target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::U(theta, 0.0, 0.0), &[], &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn cry(mut self, theta: f32, controls: &[usize], target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::U(theta, 0.0, 0.0), controls, &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn rz(mut self, theta: f32, target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::U(0.0, 0.0, theta), &[], &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    pub fn crz(mut self, theta: f32, controls: &[usize], target: usize) -> Self {
        self.instructions.push(B::from_pure(
            Gate::new(GateType::U(0.0, 0.0, theta), controls, &[target])
                .and_then(|g| g.check_qubits(self.n_qubits))
                .unwrap()
                .into(),
        ));
        self
    }

    /// Appends a circuit implementing the quantum Fourier transform.
    /// Targets are normally specified in order of least significance,
    /// for example [0,1,2,3,4].
    pub fn qft(mut self, targets: &[usize]) -> Self {
        /* This implementation is taken from Mike & Ike chapter 5.1.
         * Note that, due to our chosen convention, the circuit will
         * be the same as figure 5.1 but "upside down".
         * */

        let n = targets.len();

        for i in (0..n).rev() {
            self = self.h(targets[i]);

            let mut control: isize = i as isize - 1;
            for k in 2..(i + 2) {
                let theta = PI / (1 << (k - 1)) as f32;
                self = self.crz(theta, &[targets[control as usize]], targets[i]);
                control -= 1;
            }
        }

        // Reverse order of qubits. (not shown in figure 5.1)
        for i in 0..(n >> 1) {
            self = self.swap(targets[i], targets[n - 1 - i]);
        }
        self
    }

    /// Appends a circuit implementing a quantum oracle for a given classical function.
    /// `input_qubits` are specified in order of least significance, for example [0,1,2,3,4].
    pub fn oracle(
        self,
        input_qubits: &[usize],
        target: usize,
        classic_fn: impl Fn(usize) -> bool,
    ) -> Self {
        let truth_table = fn_to_truth_table(&classic_fn, input_qubits.len());
        let anf_coefs = truth_table_to_anf_coefs(truth_table);
        append_oracle(self, input_qubits, target, anf_coefs)
    }

    // Breakpoint

    pub fn breakpoint(mut self) -> Self {
        self.breakpoints.insert_or_enable(self.instructions.len());
        self
    }

    pub fn breakpoint_at(&self, pc: &CircuitPc) -> Option<&Breakpoint> {
        if let Some((name, pc)) = pc.next_sub_pc() {
            self.sub_circuit(name).breakpoint_at(pc)
        } else {
            self.breakpoints.get(pc.pc())
        }
    }

    pub fn enabled_breakpoint_at(&self, pc: &CircuitPc) -> bool {
        self.breakpoint_at(pc)
            .map(Breakpoint::enabled)
            .unwrap_or(false)
    }

    pub fn insert_breakpoint(&mut self, pc: &CircuitPc) -> IEBreakpoint {
        if let Some((name, pc)) = pc.next_sub_pc() {
            self.sub_circuit_mut(name).insert_breakpoint(pc)
        } else {
            self.breakpoints.insert_or_enable(pc.pc())
        }
    }

    pub fn enable_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        if let Some((name, pc)) = pc.next_sub_pc() {
            self.sub_circuit_mut(name).enable_breakpoint(pc)
        } else {
            self.breakpoints.enable(pc.pc())
        }
    }

    pub fn disable_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        if let Some((name, pc)) = pc.next_sub_pc() {
            self.sub_circuit_mut(name).disable_breakpoint(pc)
        } else {
            self.breakpoints.disable(pc.pc())
        }
    }

    pub fn delete_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        if let Some((name, pc)) = pc.next_sub_pc() {
            self.sub_circuit_mut(name).delete_breakpoint(pc)
        } else {
            self.breakpoints.delete(pc.pc())
        }
    }

    // Sub circuits

    pub fn new_sub_circuit<S: Into<String>>(mut self, name: S, pure_circuit: Circuit) -> Self {
        if self
            .sub_circuits
            .insert(name.into(), pure_circuit)
            .is_some()
        {
            log::warn!("Inserted sub circuit replaced an already defined circuit")
        }
        self
    }

    /// ## Returns
    /// If `circuit_pc` is pointing at a sub circuit, that sub cirucit will be returned.
    ///
    /// Otherwise if `circuit_pc` points at main, then `None` will be returned
    ///
    /// ## Panics
    /// Panics if `circuit_pc` points at an invalid `sub_circuit`
    pub fn current_sub_circuit(&self, circuit_pc: &CircuitPc) -> Option<&Circuit> {
        if let Some((name, pc)) = circuit_pc.next_sub_pc() {
            Some(self.sub_circuits[name].current_circuit(pc))
        } else {
            None
        }
    }

    /// ## Returns
    /// `Circuit` specified with `name`
    /// ## Panics
    /// Panics if `name` has not been registered on the current circuit
    pub fn sub_circuit(&self, name: &str) -> &Circuit {
        &self.sub_circuits[name]
    }

    /// Private for now
    fn sub_circuit_mut(&mut self, name: &str) -> &mut Circuit {
        match self.sub_circuits.get_mut(name) {
            Some(v) => v,
            None => panic!("Trying to access unregistered sub circuit {}", name),
        }
    }

    /// ## Arguments
    ///  - `name`: Name of registered sub circuit
    ///  - `lsq`: Least significant qubit that the specified sub circuit will be acting on
    pub fn call<S: Into<String>>(self, name: S, lsq: usize) -> Self {
        self.ccall(name, lsq, Default::default())
    }

    /// ## Arguments
    ///  - `name`: Name of registered sub circuit
    ///  - `lsq`: Least significant qubit that the specified sub circuit will be acting on
    ///  - `ctrl`: Control bits for sub circuit
    pub fn ccall<S: Into<String>>(mut self, name: S, lsq: usize, controls: &[usize]) -> Self {
        let name = name.into();
        if !self.sub_circuits.contains_key(&name) {
            panic!("No registered sub circuit with the name {}", name)
        } else if lsq + self.sub_circuits[&name].n_qubits() > self.n_qubits() {
            panic!(
                "Qubit overflow: Sub circuit \"{}\" and {} qubits with lsq: {} overflows the current qubit capacity of {}",
                name,
                self.sub_circuits[&name].n_qubits(),
                lsq,
                self.n_qubits()
            );
        }
        self.instructions.push(B::from_pure(PureInstruction::Call(
            name,
            lsq,
            QBits::from_indices(controls),
        )));
        self
    }

    /// ## Arguments
    ///  - `name`: Name of registered sub circuit
    ///  - `pure_circuit`: Definition of specified circuit
    ///  - `lsq`: Least significant qubit that the specified sub circuit will be acting on
    pub fn call_new<S: Into<String>>(self, name: S, pure_circuit: Circuit, lsq: usize) -> Self {
        self.ccall_new(name, pure_circuit, lsq, Default::default())
    }

    /// ## Arguments
    ///  - `name`: Name of registered sub circuit
    ///  - `pure_circuit`: Definition of specified circuit
    ///  - `lsq`: Least significant qubit that the specified sub circuit will be acting on
    ///  - `ctrl`: Control bits for sub circuit
    pub fn ccall_new<S: Into<String>>(
        self,
        name: S,
        pure_circuit: Circuit,
        lsq: usize,
        controls: &[usize],
    ) -> Self {
        let name = name.into();
        self.new_sub_circuit(name.clone(), pure_circuit)
            .ccall(name, lsq, controls)
    }
}

impl<B: CircuitBehaviour> Circuit<B>
where
    Self: Into<Circuit<HybridCircuit>>,
{
    pub fn new_reg<S: Into<String>>(self, name: S, size: usize) -> Circuit<HybridCircuit> {
        let mut ret_self = self.into();
        ret_self.registers.insert(name.into(), size);
        ret_self
    }

    // Classical instructions

    pub fn measure_bit<S: Into<String>>(
        self,
        target: usize,
        reg: (S, usize),
    ) -> Circuit<HybridCircuit> {
        let mut ret_self = self.into();
        ret_self
            .instructions
            .push(Instruction::MeasureBit(target, (reg.0.into(), reg.1)));
        ret_self
    }

    /// Measure multiple bits into a register
    ///
    /// Example:
    /// ```ignore
    /// measure_bits(&[2,1,3], "reg")
    /// // Is equivalent to
    /// measure_bit(2, ("reg", 0))
    /// measure_bit(1, ("reg", 1))
    /// measure_bit(3, ("reg", 2))
    /// ```
    pub fn measure_bits<S: Into<String>>(
        self,
        targets: &[usize],
        reg: S,
    ) -> Circuit<HybridCircuit> {
        let mut ret_self = self.into();
        let reg = reg.into();
        for (i, target) in targets.iter().enumerate() {
            ret_self = ret_self.measure_bit(*target, (&reg, i))
        }
        ret_self
    }

    pub fn measure<S: Into<String>>(self, reg: S) -> Circuit<HybridCircuit> {
        let mut ret_self = self.into();
        ret_self
            .instructions
            .push(Instruction::MeasureAll(reg.into()));
        ret_self
    }

    pub fn jump<S: Into<String>>(self, label: S) -> Circuit<HybridCircuit> {
        let mut ret_self = self.into();

        let circuit_pc = match ret_self.try_to_resolve_label(label.into()) {
            Some(circuit_pc) => circuit_pc,
            None => 0, // Placeholder pc
        };

        ret_self.instructions.push(Instruction::Jump(circuit_pc));
        ret_self
    }

    pub fn jump_if<T: Into<BoolExpr>, S: Into<String>>(
        self,
        expr: T,
        label: S,
    ) -> Circuit<HybridCircuit> {
        let mut ret_self = self.into();

        let circuit_pc = match ret_self.try_to_resolve_label(label.into()) {
            Some(circuit_pc) => circuit_pc,
            None => 0, // Placeholder pc
        };

        ret_self
            .instructions
            .push(Instruction::JumpIf(expr.into(), circuit_pc));
        ret_self
    }

    /// Conditionally apply whichever instruction that comes after
    pub fn apply_if<T: Into<BoolExpr>>(self, expr: T) -> Circuit<HybridCircuit> {
        let mut ret_self = self.into();
        ret_self.instructions.push(Instruction::JumpIf(
            !expr.into(),
            ret_self.instructions.len() + 2,
        ));
        ret_self
    }

    pub fn reset(self, target: usize) -> Circuit<HybridCircuit> {
        self.new_reg("_reset", 1)
            .measure_bit(target, ("_reset", 0))
            .apply_if(BitExpr::Reg("_reset".to_owned()).eq(1))
            .x(target)
    }

    // takes register nr directly for now
    pub fn assign<S: Into<String>, T: Into<BitExpr>>(
        self,
        reg: S,
        expr: T,
    ) -> Circuit<HybridCircuit> {
        let mut ret_self = self.into();
        let reg = reg.into();
        if !ret_self.registers.contains_key(&reg) {
            panic!(
                "Tried to assign to nonexistent register with name '{}'.",
                &reg
            )
        }
        ret_self
            .instructions
            .push(Instruction::Assign(expr.into(), reg));
        ret_self
    }

    // Label
    pub fn label<S: Into<String>>(self, label: S) -> Circuit<HybridCircuit> {
        let mut ret_self = self.into();
        let pc = ret_self.instructions.len();
        let label = label.into();

        if let Some(idx) = ret_self.labels.get(&label) {
            panic!("Label '{label}' was already defined on instruction row {idx}")
        }

        ret_self.labels.insert(label, pc);

        // After a new label has been added we try to resolve unresolved labels and patch instructions
        ret_self.try_to_patch_instructions();
        ret_self
    }
}

// Helper functions for hybrid circuits
impl Circuit<HybridCircuit> {
    fn try_to_patch_instructions(&mut self) {
        let mut to_remove = Vec::new();

        // 2. Patch instructions
        for (label, pc) in self.unresolved_labels.clone() {
            let Some(&resolved_pc) = self.labels.get(&label) else {
                continue;
            };

            let inst = &mut self.instructions[pc];

            match inst {
                Instruction::Jump(jump_pc) => *jump_pc = resolved_pc,
                Instruction::JumpIf(_expr, jump_pc) => *jump_pc = resolved_pc,
                _ => continue,
            };

            to_remove.push((label.clone(), pc));
        }

        // 3. Remove resolved labels after patching
        self.unresolved_labels
            .retain(|(label, idx)| !to_remove.contains(&(label.clone(), *idx)));
    }

    fn try_to_resolve_label(&mut self, label: String) -> Option<usize> {
        if let Some(&pc) = self.labels.get(&label) {
            return Some(pc);
        }

        // 1. If label doesnt exist, add it to list of unresolved labels with accompanying instruction index
        let pair = (label, self.instructions.len());
        self.unresolved_labels.push(pair);
        None
    }
}

impl Into<Circuit<HybridCircuit>> for Circuit<PureCircuit> {
    fn into(self) -> Circuit<HybridCircuit> {
        Circuit {
            instructions: self
                .instructions
                .into_iter()
                .map(HybridCircuit::from_pure)
                .collect(),
            n_qubits: self.n_qubits,
            labels: self.labels,
            unresolved_labels: self.unresolved_labels,
            breakpoints: self.breakpoints,
            registers: self.registers,
            sub_circuits: self.sub_circuits,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HybridCircuit;
impl CircuitBehaviour for HybridCircuit {
    type InstructionTy = Instruction;

    fn from_pure(instruction: PureInstruction) -> Self::InstructionTy {
        Instruction::from(instruction)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PureCircuit;
impl CircuitBehaviour for PureCircuit {
    type InstructionTy = PureInstruction;

    fn from_pure(instruction: PureInstruction) -> Self::InstructionTy {
        instruction
    }
}

pub trait CircuitBehaviour {
    type InstructionTy;
    fn from_pure(instruction: PureInstruction) -> Self::InstructionTy;
}

pub struct FlatCircuit<'a, B: CircuitBehaviour> {
    circuit: &'a Circuit<B>,
    pc: CircuitPc,
}

impl<'a> Iterator for FlatCircuit<'a, PureCircuit> {
    type Item = PureInstruction;

    fn next(&mut self) -> Option<Self::Item> {
        let inst = self.circuit.instruction(&self.pc);
        match inst {
            Some(PureInstruction::Call(name, lsq, ctrl)) => {
                self.pc.jump_and_link(name, lsq, ctrl);
                // Take next instruction inside sub circuit
                self.next()
            }
            Some(inst) => {
                self.pc.increment();
                Some(inst)
            }
            None => {
                // Try to return
                if self.pc.ret() {
                    // Could return: Return next
                    self.next()
                } else {
                    // Could not return: end of circuit
                    None
                }
            }
        }
    }
}

impl<'a> Iterator for FlatCircuit<'a, HybridCircuit> {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        let inst = self.circuit.instruction(&self.pc);
        match inst {
            Some(Instruction::Call(name, lsq, ctrl)) => {
                self.pc.jump_and_link(name, lsq, ctrl);
                // Take next instruction inside sub circuit
                self.next()
            }
            Some(inst) => {
                self.pc.increment();
                Some(inst)
            }
            None => {
                // Try to return
                if self.pc.ret() {
                    // Could return: Return next
                    self.next()
                } else {
                    // Could not return: end of circuit
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cart,
        circuit::Circuit,
        ext::{equal_matrix_c, equal_state_c, expand_matrix_from_gate},
        instruction::{Instruction, PureInstruction},
        simulator::{Buildable, Simulator},
        sv_simulator::StateVectorSimulator,
    };
    use nalgebra::{Complex, DMatrix, dvector};
    use std::panic::catch_unwind;
    fn concat_circuits(circuit1: &Circuit, circuit2: &Circuit) -> Circuit {
        let mut circuit_tot = Circuit::new(std::cmp::max(circuit1.n_qubits(), circuit2.n_qubits()));
        circuit_tot.instructions =
            [circuit1.instructions.clone(), circuit2.instructions.clone()].concat();
        circuit_tot
    }
    #[test]
    fn error_test() {
        let c = Circuit::new(5);
        assert!(catch_unwind(|| c.clone().h(0)).is_ok());
        assert!(catch_unwind(|| c.clone().ch(&[3, 1, 2], 0)).is_ok());
        assert!(catch_unwind(|| c.clone().ch(&[3, 1, 3], 0)).is_ok());

        assert!(catch_unwind(|| c.clone().ch(&[3, 1, 0], 0)).is_err()); // Overlap
        assert!(catch_unwind(|| c.clone().ch(&[3, 1, 0], 5)).is_err()); // Target out of bounds
        assert!(catch_unwind(|| c.clone().ch(&[3, 1, 5], 2)).is_err()); // Control out of bounds
    }
    #[test]
    fn inverse_test() {
        let circ = Circuit::new(5)
            .h(0)
            .h(1)
            .h(3)
            .x(0)
            .y(1)
            .z(2)
            .s(4)
            .cx(&[0], 1)
            .cx(&[4], 1)
            .u(23.3, 34.5, 56.1, 0)
            .cu(1.0, 22.2, 0.1, &[4], 2)
            .swap(3, 4)
            .cswap(&[0], 1, 2);
        let circ_and_inv = concat_circuits(&circ, &circ.inverse());
        let dim = 1 << 5;
        let id = DMatrix::<Complex<f32>>::identity(dim, dim);
        let mut res: DMatrix<Complex<f32>> = id.clone();
        for instruction in circ_and_inv.instructions() {
            if let PureInstruction::Gate(gate) = instruction {
                res = expand_matrix_from_gate(gate, 5) * res;
            }
        }
        assert!(equal_matrix_c(&id, &res, 5, 0.001));
    }
    #[test]
    fn qft_test() {
        let mut sim = StateVectorSimulator::build(Circuit::new(4).x(0).y(1).z(2).h(3).call_new(
            "QFT",
            Circuit::new_qft(4),
            0,
        ))
        .unwrap();
        sim.run();

        let expected_vec = dvector![
            cart!(0.0, 0.35355),  // |0000>
            cart!(0.0),           // |0001>
            cart!(-0.25, -0.25),  // |0010>
            cart!(0.0),           // |0011>
            cart!(0.35355, 0.0),  // |0100>
            cart!(0.0),           // |0101>
            cart!(-0.25, 0.25),   // |0110>
            cart!(0.0),           // |0111>
            cart!(0.0, -0.35355), // |1000>
            cart!(0.0),           // |1001>
            cart!(0.25, 0.25),    // |1010>
            cart!(0.0),           // |1011>
            cart!(-0.35355, 0.0), // |1100>
            cart!(0.0),           // |1101>
            cart!(0.25, -0.25),   // |1110>
            cart!(0.0),           // |1111>
        ];
        assert!(equal_state_c(&expected_vec, sim.state(), 4, 0.001));
    }

    #[test]
    fn unresolved_labels_are_patched_without_duplication() {
        let circuit = Circuit::new(1)
            .jump("target")
            .label("other")
            .x(0)
            .label("target");

        assert!(!circuit.has_unresolved_labels());
        assert_eq!(circuit.instructions()[0], Instruction::Jump(2));
    }

    #[test]
    fn flatten() {
        let sub1 = Circuit::new(2).h(0).h(1);
        let sub2 = Circuit::new(4)
            .h(0)
            .call_new("sub1", sub1, 0)
            .h(2)
            .call("sub1", 2);

        let circuit = Circuit::new(6)
            .new_reg("tt", 6)
            .h(0)
            .call_new("sub2", sub2, 0)
            .measure("tt")
            .call("sub2", 2)
            .h(2);

        let correct = Circuit::new(6)
            .new_reg("tt", 6)
            // main
            .h(0) // 0
            // sub2 @ 0
            .h(0) // 1
            // sub1 @ 0
            .h(0) // 2
            .h(1) // 3
            // end
            .h(2) // 4
            // sub1 @ 2
            .h(2) // 5
            .h(3) // 6
            // end
            // end
            .measure("tt") // 7
            // sub2 @ 2
            .h(2) // 8
            // sub1 @ 2
            .h(2) // 9
            .h(3) // 10
            // end
            .h(4) // 11
            // sub1 @ 4
            .h(4) // 12
            .h(5) // 13
            // end
            // end
            .h(2)
            .instructions;

        let mut as_flat1 = circuit.as_flat();
        for i in 0..correct.len() {
            assert_eq!(as_flat1.next(), Some(correct[i].clone()))
        }
        assert!(as_flat1.next().is_none());
    }
}
