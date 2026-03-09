pub mod breakpoint;
pub mod pc;

use std::{
    collections::{HashMap, HashSet},
};

use crate::{
    circuit::{
        breakpoint::{Breakpoint, BreakpointList, IEBreakpoint},
        pc::CircuitPc,
    },
    expr_dsl::Expr,
    gate::{Gate, GateType, QBits},
    instruction::Instruction,
};
mod qasm_parse;

use log::{trace, warn};
use oq3_syntax::{SourceFile, ast::AstNode};
use std::fs::read_to_string;

pub use qasm_parse::*;

#[derive(Debug, Clone)]
pub struct Circuit {
    instructions: Vec<Instruction>,
    n_qubits: usize,
    labels: HashMap<String, usize>,
    unresolved_labels: Vec<(String, usize)>,
    breakpoints: BreakpointList,
    registers: HashSet<String>,
}

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            instructions: Vec::<Instruction>::default(),
            n_qubits: n_qubits,
            registers: HashSet::new(),
            labels: HashMap::new(),
            unresolved_labels: Vec::new(),
            breakpoints: Default::default(),
        }
    }

    pub fn new_reg<I: Into<String>>(mut self, name: I) -> Self {
        self.registers.insert(name.into());
        self
    }

    pub fn valid_pc(&self, circuit_pc: &CircuitPc) -> bool {
        circuit_pc.pc() <= self.instructions().len()
    }

    pub fn instruction(&self, circuit_pc: &CircuitPc) -> Option<Instruction> {
        match self.instructions().get(circuit_pc.pc()) {
            Some(Instruction::Gate(gate)) => {
                Some(Instruction::Gate(gate.clone() << circuit_pc.lsq()))
            }
            Some(Instruction::Measurement(targets, register)) => Some(Instruction::Measurement(
                *targets << circuit_pc.lsq(),
                register.clone(),
            )),
            rst => rst.cloned(),
        }
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn registers(&self) -> &HashSet<String> {
        &self.registers
    }

    pub fn has_unresolved_labels(&self) -> bool {
        !self.unresolved_labels.is_empty()
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

    // Builder methods

    pub fn x(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::X, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn y(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Y, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn z(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Z, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn hadamard(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::H, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::X, &[control], &[target]).unwrap(),
        ));
        self
    }

    pub fn swap(mut self, target1: usize, target2: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::SWAP, &[], &[target1, target2]).unwrap(),
        ));
        self
    }

    pub fn fredkin(mut self, control: usize, target1: usize, target2: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::SWAP, &[control], &[target1, target2]).unwrap(),
        ));
        self
    }

    pub fn u(mut self, theta: f64, phi: f64, lambda: f64, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, phi, lambda), &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cu(mut self, theta: f64, phi: f64, lambda: f64, control: usize, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, phi, lambda), &[control], &[target]).unwrap(),
        ));
        self
    }

    pub fn s(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::S, &[], &[target]).unwrap(),
        ));
        self
    }

    // Classical instructions

    pub fn measure_bit(mut self, target: usize, reg: &str) -> Self {
        self.instructions.push(Instruction::Measurement(
            QBits::from_bitstring(1 << target),
            reg.to_owned(),
        ));
        self
    }

    pub fn measure(mut self, targets: &[usize], reg: &str) -> Self {
        self.instructions.push(Instruction::Measurement(
            QBits::from_indices(targets),
            reg.to_owned(),
        ));
        self
    }

    pub fn jump(mut self, label: String) -> Self {
        let circuit_pc = match self.try_to_resolve_label(label) {
            Some(circuit_pc) => circuit_pc.clone(),
            None => 0, // Placeholder pc
        };

        self.instructions.push(Instruction::Jump(circuit_pc));
        self
    }

    pub fn jump_if(mut self, expr: Expr, label: String) -> Self {
        let circuit_pc = match self.try_to_resolve_label(label) {
            Some(circuit_pc) => circuit_pc.clone(),
            None => 0, // Placeholder pc
        };

        self.instructions
            .push(Instruction::JumpIf(expr, circuit_pc));
        self
    }

    /// Conditionally apply whichever instruction that comes after
    pub fn apply_if(mut self, expr: Expr) -> Self {
        self.instructions
            .push(Instruction::JumpIf(!expr, self.instructions.len() + 2));
        self
    }

    pub fn reset(self, target: usize) -> Self {
        self.new_reg("_reset")
            .measure_bit(target, "_reset")
            .apply_if(Expr::Reg("_reset".to_owned()).eq(1))
            .x(target)
    }

    // takes register nr directly for now
    pub fn assign(mut self, reg: String, expr: Expr) -> Self {
        if !self.registers.contains(&reg) {
            panic!(
                "Tried to assign to nonexistent register with name '{}'.",
                reg
            )
        }
        self.instructions.push(Instruction::Assign(expr, reg));
        self
    }

    /// Controlled R_k
    pub fn crk(mut self, k: usize, control: usize, target: usize) -> Self {
        let pow2_inv = 1.0 / (1 << (k - 1)) as f64;
        self.instructions.push(Instruction::Gate(
            Gate::new(
                GateType::U(0.0, 0.0, pow2_inv * std::f64::consts::PI),
                &[control],
                &[target],
            )
            .unwrap(),
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
            self = self.hadamard(targets[i]);

            let mut control: isize = i as isize - 1;
            for k in 2..(i + 2) {
                self = self.crk(k, targets[control as usize], targets[i]);
                control -= 1;
            }
        }

        // Reverse order of qubits. (not shown in figure 5.1)
        for i in 0..(n >> 1) {
            self = self.swap(targets[i], targets[n - 1 - i]);
        }
        self
    }

    // Label
    pub fn label(mut self, label: String) -> Self {
        let pc = self.instructions.len();

        if let Some(idx) = self.labels.get(&label) {
            panic!("Label '{label}' was already defined on instruction row {idx}")
        }

        self.labels.insert(label, pc);

        // After a new label has been added we try to resolve unresolved labels and patch instructions
        self.try_to_patch_instructions();
        self
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

    fn try_to_patch_instructions(&mut self) {
        let mut to_remove = Vec::new();

        // 2. Patch instructions
        for (label, pc) in self.unresolved_labels.clone() {
            let Some(resolved_pc) = self.try_to_resolve_label(label.clone()) else {
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

    // Breakpoint

    pub fn breakpoint(mut self) -> Self {
        self.breakpoints.insert_or_enable(self.instructions.len());
        self
    }

    pub fn next_break(&self, pc: &CircuitPc) -> Option<(CircuitPc, bool)> {
        let brk = self.breakpoints.next_break(pc.pc())?;
        Some((CircuitPc::new(brk.pc()), brk.enabled()))
    }

    pub fn next_enabled_break(&self, pc: &CircuitPc) -> Option<CircuitPc> {
        while let Some((pc, enabled)) = self.next_break(pc) {
            if enabled {
                return Some(pc);
            }
        }
        None
    }

    pub fn breakpoint_at(&self, pc: &CircuitPc) -> Option<&Breakpoint> {
        self.breakpoints.get(pc.pc())
    }

    pub fn enabled_breakpoint_at(&self, pc: &CircuitPc) -> bool {
        self.breakpoint_at(pc)
            .map(Breakpoint::enabled)
            .unwrap_or(false)
    }

    pub fn insert_breakpoint(&mut self, pc: &CircuitPc) -> IEBreakpoint {
        self.breakpoints.insert_or_enable(pc.pc())
    }

    pub fn enable_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        self.breakpoints.enable(pc.pc())
    }

    pub fn disable_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        self.breakpoints.disable(pc.pc())
    }

    pub fn delete_breakpoint(&mut self, pc: &CircuitPc) -> bool {
            self.breakpoints.delete(pc.pc())
    }
}
