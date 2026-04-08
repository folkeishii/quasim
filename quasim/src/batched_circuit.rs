use std::collections::{BTreeMap, HashSet};

use crate::{
    circuit::{Circuit, CircuitBehaviour, HybridCircuit},
    gate::QBits,
    gate_batcher::{BatchCommand, GateBatchData, GateBatcher},
    instruction::{Instruction, PureInstruction},
};

#[derive(Debug, Clone)]
pub struct BatchedCircuit {
    batcher: GateBatcher,
    instruction_lookup: BTreeMap<usize, BatchedCircuitOp>,
}

#[derive(Debug, Clone)]
pub enum BatchedCircuitOp {
    BatchCommands(Vec<BatchCommand>),
    Instruction(Instruction),
}

impl BatchedCircuit {
    const DEFAULT_MAX_TARGET_QUBITS: usize = 3;

    pub fn from_circuit<B>(circuit: Circuit<B>, max_target_qubits: usize) -> Self
    where
        B: CircuitBehaviour,
        Circuit<B>: Into<Circuit<HybridCircuit>>,
    {
        let mut batched_circuit = Self {
            batcher: GateBatcher::new(max_target_qubits),
            instruction_lookup: BTreeMap::new(),
        };

        let circuit = circuit.into();

        let mut batch_start_inst_index = 0;
        let flat_instructions = flattened_instructions(&circuit);
        let jump_targets = collect_jump_targets(&flat_instructions);

        for (inst_index, inst) in flat_instructions.into_iter().enumerate() {
            if jump_targets.contains(&inst_index) {
                batched_circuit.flush_batches(batch_start_inst_index);
                batch_start_inst_index = inst_index;
            }

            match inst {
                Instruction::Gate(gate) => {
                    batched_circuit.batcher.add_gate(&gate);
                }
                _ => {
                    batched_circuit.flush_batches(batch_start_inst_index);
                    batched_circuit
                        .instruction_lookup
                        .insert(inst_index, BatchedCircuitOp::Instruction(inst));
                    batch_start_inst_index = inst_index + 1;
                }
            }
        }

        batched_circuit.flush_batches(batch_start_inst_index);

        batched_circuit
    }

    pub fn data(&self) -> &GateBatchData {
        &self.batcher.data()
    }

    pub fn operation(&self, index: usize) -> Option<&BatchedCircuitOp> {
        self.instruction_lookup
            .range(index..)
            .next()
            .and_then(|(_, op)| Some(op))
    }

    fn flush_batches(&mut self, to_inst_index: usize) {
        let batch_commands = self.batcher.flush_batches();

        if batch_commands.is_empty() {
            return;
        }

        self.instruction_lookup.insert(
            to_inst_index,
            BatchedCircuitOp::BatchCommands(batch_commands),
        );
    }
}

impl<B> From<Circuit<B>> for BatchedCircuit
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
{
    fn from(value: Circuit<B>) -> Self {
        BatchedCircuit::from_circuit(value, BatchedCircuit::DEFAULT_MAX_TARGET_QUBITS)
    }
}

fn flattened_instructions(circuit: &Circuit<HybridCircuit>) -> Vec<Instruction> {
    let mut flat_instructions = Vec::new();
    let mut top_level_pc_to_flat_pc = Vec::with_capacity(circuit.instructions().len() + 1);

    // Hybrid jump targets are resolved against top-level instruction rows.
    // Once sub-circuit calls are flattened, those rows need to be remapped to
    // their new flat instruction indices before the GPU executor can jump safely.
    for instruction in circuit.instructions() {
        top_level_pc_to_flat_pc.push(flat_instructions.len());
        push_flat_instruction(circuit, instruction, &mut flat_instructions);
    }

    top_level_pc_to_flat_pc.push(flat_instructions.len());
    remap_jump_targets(&mut flat_instructions, &top_level_pc_to_flat_pc);

    flat_instructions
}

fn push_flat_instruction(
    circuit: &Circuit<HybridCircuit>,
    instruction: &Instruction,
    flat_instructions: &mut Vec<Instruction>,
) {
    match instruction {
        Instruction::Call(name, lsq, ctrl) => {
            push_flat_sub_circuit(circuit.sub_circuit(name), *lsq, *ctrl, flat_instructions)
        }
        _ => flat_instructions.push(instruction.clone()),
    }
}

fn push_flat_sub_circuit(
    sub_circuit: &Circuit,
    lsq: usize,
    ctrl: QBits,
    flat_instructions: &mut Vec<Instruction>,
) {
    for instruction in sub_circuit.as_flat() {
        let PureInstruction::Gate(mut gate) = instruction else {
            unreachable!("flattened pure sub-circuits should only yield gates");
        };
        gate = gate << lsq;
        *gate.control_mut() |= ctrl;
        flat_instructions.push(Instruction::Gate(gate));
    }
}

fn remap_jump_targets(instructions: &mut [Instruction], top_level_pc_to_flat_pc: &[usize]) {
    for instruction in instructions {
        let target = match instruction {
            Instruction::Jump(target) => target,
            Instruction::JumpIf(_, target) => target,
            _ => continue,
        };

        *target = *top_level_pc_to_flat_pc
            .get(*target)
            .expect("jump target should refer to a valid top-level pc");
    }
}

fn collect_jump_targets(instructions: &[Instruction]) -> HashSet<usize> {
    let mut targets = HashSet::new();

    for inst in instructions {
        match inst {
            Instruction::Jump(pc) => targets.insert(*pc),
            Instruction::JumpIf(_, pc) => targets.insert(*pc),
            _ => false,
        };
    }

    targets
}

#[cfg(test)]
mod tests {
    use crate::{circuit::Circuit, expr_dsl::expr_helpers::r, instruction::Instruction};

    use super::{BatchedCircuit, BatchedCircuitOp, flattened_instructions};

    #[test]
    fn test_flatten_jump_target_remapping() {
        let sub = Circuit::new(2).h(0).h(1);
        let circuit = Circuit::new(3)
            .new_reg("flag", 1)
            .call_new("sub", sub, 0)
            .jump("skip")
            .x(2)
            .label("skip")
            .jump_if(r("flag").eq(0), "done")
            .x(1)
            .label("done");

        let flat = flattened_instructions(&circuit);

        assert_eq!(flat.len(), 6);
        assert!(matches!(flat[2], Instruction::Jump(4)));
        assert!(matches!(flat[4], Instruction::JumpIf(_, 6)));

        let batched = BatchedCircuit::from(circuit);

        assert!(matches!(
            batched.operation(2),
            Some(BatchedCircuitOp::Instruction(Instruction::Jump(4)))
        ));
        assert!(matches!(
            batched.operation(4),
            Some(BatchedCircuitOp::Instruction(Instruction::JumpIf(_, 6)))
        ));
    }
}
