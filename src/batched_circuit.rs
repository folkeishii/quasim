use std::collections::BTreeMap;

use crate::{
    circuit::{Circuit, CircuitBehaviour, HybridCircuit},
    gate_batcher::{BatchCommand, GateBatchData, GateBatcher},
    instruction::Instruction,
};

#[derive(Clone)]
pub struct BatchedCircuit {
    batcher: GateBatcher,
    instruction_lookup: BTreeMap<usize, BatchedCircuitOp>,
    data: GateBatchData,
}

#[derive(Clone)]
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
            data: GateBatchData::new(),
        };

        let circuit = circuit.into();

        let mut batch_start_inst_index = 0;

        for (inst_index, inst) in circuit.instructions().iter().enumerate() {
            match inst {
                Instruction::Gate(gate) => {
                    batched_circuit.batcher.add_gate(gate);
                }
                _ => {
                    batched_circuit.flush_batches(batch_start_inst_index);
                    batched_circuit
                        .instruction_lookup
                        .insert(inst_index, BatchedCircuitOp::Instruction(inst.clone()));
                    batch_start_inst_index = inst_index + 1;
                }
            }
        }

        batched_circuit
    }

    pub fn data(&self) -> &GateBatchData {
        &self.data
    }

    pub fn operation(&self, index: usize) -> Option<&BatchedCircuitOp> {
        self.instruction_lookup
            .range(index..)
            .next()
            .and_then(|(_, op)| Some(op))
    }

    fn flush_batches(&mut self, to_inst_index: usize) {
        let batch_data = self.batcher.flush_batches();
        self.instruction_lookup.insert(
            to_inst_index,
            BatchedCircuitOp::BatchCommands(batch_data.commands().to_vec()),
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
