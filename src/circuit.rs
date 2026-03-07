use std::{
    borrow::{Borrow, Cow},
    collections::{HashMap, HashSet},
    f64::consts::PI,
    fmt::{Display, Write},
    ops::{Deref, DerefMut},
};
pub mod breakpoint;
pub mod pc;

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
    // Local to main
    instructions: Vec<Instruction>,
    n_qubits: usize,
    labels: HashMap<String, usize>,
    unresolved_labels: Vec<(String, usize)>,
    breakpoints: BreakpointList,

    // Global for all sub circuits
    registers: HashSet<String>,
    sub_circuits: HashMap<String, SubCircuit>,
    unresolved_sub_circuits: HashSet<String>,
}

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            instructions: Vec::<Instruction>::default(),
            n_qubits: n_qubits,
            registers: HashSet::new(),
            labels: HashMap::new(),
            unresolved_labels: Vec::new(),
            sub_circuits: HashMap::new(),
            unresolved_sub_circuits: HashSet::new(),
            breakpoints: Default::default(),
        }
    }

    pub fn new_reg<I: Into<String>>(mut self, name: I) -> Self {
        self.registers.insert(name.into());
        self
    }

    pub fn valid_pc(&self, circuit_pc: &CircuitPc) -> bool {
        circuit_pc.pc()
            <= self
                .instructions(circuit_pc.sub_circuit().map(AsRef::as_ref))
                .len()
    }

    pub fn instruction(&self, circuit_pc: &CircuitPc) -> Option<Instruction> {
        match self
            .instructions(circuit_pc.sub_circuit().map(AsRef::as_ref))
            .get(circuit_pc.pc())
        {
            Some(Instruction::Gate(gate)) => {
                Some(Instruction::Gate(gate.clone() << circuit_pc.lsq()))
            }
            Some(Instruction::Measurement(targets, register)) => Some(Instruction::Measurement(
                *targets << circuit_pc.lsq(),
                register.clone(),
            )),
            Some(Instruction::Call(sc, lsq)) => {
                Some(Instruction::Call(sc.clone(), lsq + circuit_pc.lsq()))
            }
            rst => rst.cloned(),
        }
    }

    pub fn instructions(&self, sub_circuit: Option<&str>) -> &[Instruction] {
        if let Some(sc) = sub_circuit {
            self.sub_circuit(sc).instructions()
        } else {
            &self.instructions
        }
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

    pub fn cx(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::X, controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn y(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Y, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cy(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Y, controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn z(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Z, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cz(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::Z, controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn h(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::H, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn ch(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::H, controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn swap(mut self, target1: usize, target2: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::SWAP, &[], &[target1, target2]).unwrap(),
        ));
        self
    }

    pub fn cswap(mut self, controls: &[usize], target1: usize, target2: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::SWAP, controls, &[target1, target2]).unwrap(),
        ));
        self
    }

    pub fn u(mut self, theta: f64, phi: f64, lambda: f64, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, phi, lambda), &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cu(
        mut self,
        theta: f64,
        phi: f64,
        lambda: f64,
        controls: &[usize],
        target: usize,
    ) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, phi, lambda), controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn s(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::S, &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cs(mut self, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::S, controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn rx(mut self, theta: f64, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, -PI / 2.0, PI / 2.0), &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn crx(mut self, theta: f64, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, -PI / 2.0, PI / 2.0), controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn ry(mut self, theta: f64, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, 0.0, 0.0), &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn cry(mut self, theta: f64, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(theta, 0.0, 0.0), controls, &[target]).unwrap(),
        ));
        self
    }

    pub fn rz(mut self, theta: f64, target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(0.0, 0.0, theta), &[], &[target]).unwrap(),
        ));
        self
    }

    pub fn crz(mut self, theta: f64, controls: &[usize], target: usize) -> Self {
        self.instructions.push(Instruction::Gate(
            Gate::new(GateType::U(0.0, 0.0, theta), controls, &[target]).unwrap(),
        ));
        self
    }

    // Sub circuit

    pub fn sub_circuit<I: Into<String>>(mut self, name: I, first_qubit: usize) -> Self {
        self.instructions
            .push(Instruction::SubCircuit(name.into(), first_qubit));
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
                let theta = PI / (1 << (k - 1)) as f64;
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

    /// Inverts a non-hybrid circuit.
    pub fn inverse(&self) -> Self {
        let mut inverted_circuit = Circuit::new(self.n_qubits());
        for instruction in &self.instructions {
            match instruction {
                Instruction::Gate(gate) => inverted_circuit
                    .instructions
                    .push(Instruction::Gate(gate.inverse())),
                _ => panic!("Circuit is hybrid"),
            }
        }
        inverted_circuit.instructions.reverse();
        inverted_circuit
    }
    // Sub circuit

    pub fn new_sub_circuit<I: Into<String>>(mut self, name: I, circuit: Circuit) -> Self {
        let name = name.into();

        let (
            sub_circuit,
            SubCircuitPeriphs {
                registers,
                sub_circuits,
                unresolved_sub_circuits,
            },
        ) = SubCircuit::from_circuit(circuit);

        // Check if sub circuit with same name but different definition already exists
        if let Some(existing) = self.sub_circuits.get(&name)
            && sub_circuit.eq(existing)
            // Is existing a placeholder
            && !self.unresolved_sub_circuits.contains(&name)
        {
            panic!(
                "Cannot add {}: A different sub circuit with the same name is already defined",
                name
            )
        }
        self.unresolved_sub_circuits.remove(&name);

        let importing_sub_circuits = sub_circuits
            .into_iter()
            // Don't try to import unresolved circuits from sub circuit
            .filter(|(imp_sc, _)| !unresolved_sub_circuits.contains(imp_sc));

        for (imp_name, imp_val) in importing_sub_circuits {
            // Does a sub circuit with the same name exist
            let Some(exist_val) = self.sub_circuits.get(&imp_name) else {
                self.sub_circuits.insert(imp_name, imp_val);
                continue;
            };

            // It exist but is a placeholder
            if self.unresolved_sub_circuits.contains(&imp_name) {
                self.unresolved_sub_circuits.remove(&imp_name);
                self.sub_circuits.insert(imp_name, imp_val);
                continue;
            }

            // If it does exist they must have the same definition
            if !imp_val.eq(exist_val) {
                panic!(
                    "Cannot add {}: A different sub circuit with the name \"{}\" is already defined",
                    name, imp_name
                )
            }

            // Make sure that if sub circuit calls "itself"
            // they must have the same definition
            if name.eq(&imp_name) && !sub_circuit.eq(&imp_val) {
                panic!(
                    "Cannot add {}: A different sub circuit with the name \"{}\" is already defined",
                    name, imp_name
                )
            }
        }

        // Filter out unresolved sub circuits that are defined
        for imp_name in unresolved_sub_circuits {
            // Insert placeholder for unresolved circuits
            if !self.sub_circuits.contains_key(&imp_name) {
                self.sub_circuits.insert(imp_name, SubCircuit::identity());
            }
        }

        // Extend
        self.sub_circuits.insert(name, sub_circuit);
        self.registers.extend(registers);

        // Return
        self
    }

    pub fn call<I: Into<String>>(mut self, name: I, lsq: usize) -> Self {
        let name = name.into();
        if !self.sub_circuits.contains_key(&name) {
            self.sub_circuits
                .insert(name.clone(), SubCircuit::identity());
            self.unresolved_sub_circuits.insert(name.clone());
        }
        self.instructions.push(Instruction::Call(name, lsq));
        self
    }

    // Breakpoint

    pub fn breakpoint(mut self) -> Self {
        self.breakpoints.insert_or_enable(self.instructions.len());
        self
    }

    pub fn next_break(&self, pc: &CircuitPc) -> Option<(CircuitPc, bool)> {
        if let Some(sc) = pc.sub_circuit() {
            let brk = self.sub_circuits[sc].breakpoints.next_break(pc.pc())?;
            Some((pc.with_pc(brk.pc()), brk.enabled()))
        } else {
            let brk = self.breakpoints.next_break(pc.pc())?;
            Some((pc.with_pc(brk.pc()), brk.enabled()))
        }
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
        if let Some(sc) = pc.sub_circuit() {
            self.sub_circuits[sc].breakpoints.get(pc.pc())
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
        if let Some(sc) = pc.sub_circuit() {
            self.sub_circuits
                .get_mut(sc)
                .expect("pc does not point at a registered sub circuit")
                .breakpoints
                .insert_or_enable(pc.pc())
        } else {
            self.breakpoints.insert_or_enable(pc.pc())
        }
    }

    pub fn enable_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        if let Some(sc) = pc.sub_circuit() {
            self.sub_circuits
                .get_mut(sc)
                .expect("pc does not point at a registered sub circuit")
                .breakpoints
                .enable(pc.pc())
        } else {
            self.breakpoints.enable(pc.pc())
        }
    }

    pub fn disable_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        if let Some(sc) = pc.sub_circuit() {
            self.sub_circuits
                .get_mut(sc)
                .expect("pc does not point at a registered sub circuit")
                .breakpoints
                .disable(pc.pc())
        } else {
            self.breakpoints.disable(pc.pc())
        }
    }

    pub fn delete_breakpoint(&mut self, pc: &CircuitPc) -> bool {
        if let Some(sc) = pc.sub_circuit() {
            self.sub_circuits
                .get_mut(sc)
                .expect("pc does not point at a registered sub circuit")
                .breakpoints
                .delete(pc.pc())
        } else {
            self.breakpoints.delete(pc.pc())
        }
    }

    fn sub_circuit(&self, name: &str) -> &SubCircuit {
        if let Some(sub_circuit) = self.sub_circuits.get(name) {
            sub_circuit
        } else {
            panic!("Cannot access undefined sub circuit {}", name)
        }
    }

    fn sub_circuit_mut(&mut self, name: &str) -> &mut SubCircuit {
        if let Some(sub_circuit) = self.sub_circuits.get_mut(name) {
            sub_circuit
        } else {
            panic!("Cannot access undefined sub circuit {}", name)
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubCircuit {
    instructions: Vec<Instruction>,
    n_qubits: usize,
    labels: HashMap<String, usize>,
    breakpoints: BreakpointList,
}
impl SubCircuit {
    pub fn identity() -> Self {
        Self {
            instructions: Vec::with_capacity(0),
            n_qubits: 1,
            labels: HashMap::with_capacity(0),
            breakpoints: BreakpointList::default(),
        }
    }

    pub fn from_circuit(circuit: Circuit) -> (Self, SubCircuitPeriphs) {
        let Circuit {
            instructions,
            n_qubits,
            registers,
            labels,
            unresolved_labels,
            sub_circuits,
            unresolved_sub_circuits,
            breakpoints,
        } = circuit;
        if !unresolved_labels.is_empty() {
            panic!("Tried to create a sub circuit from a circuit with unresolved labels");
        }
        (
            Self {
                instructions,
                n_qubits,
                labels,
                breakpoints,
            },
            SubCircuitPeriphs {
                registers,
                sub_circuits,
                unresolved_sub_circuits,
            },
        )
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    pub fn instructions_mut(&mut self) -> &mut [Instruction] {
        &mut self.instructions
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }
}
impl PartialEq for SubCircuit {
    fn eq(&self, other: &Self) -> bool {
        self.instructions == other.instructions
    }
}

#[derive(Debug, Clone)]
pub struct SubCircuitPeriphs {
    registers: HashSet<String>,
    sub_circuits: HashMap<String, SubCircuit>,
    unresolved_sub_circuits: HashSet<String>,
}
