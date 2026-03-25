use std::collections::BTreeMap;

use crate::{
    circuit::{Circuit, HybridCircuit, PureCircuit},
    gate_batcher::{GateBatchData, GateBatcher},
    instruction::Instruction,
};

pub struct BatchedCircuit {
    batcher: GateBatcher,
    instruction_lookup: BTreeMap<usize, BatchedCircuitOp>,
    data: GateBatchData,
}

pub enum BatchedCircuitOp {
    BatchOffsets(Vec<usize>),
    Instruction(Instruction),
}

impl BatchedCircuit {
    const DEFAULT_MAX_TARGET_QUBITS: usize = 3;

    fn flush_batches(&mut self, to_inst_index: usize) {
        let batch_data = self.batcher.flush_batches();
        self.instruction_lookup.insert(
            to_inst_index,
            BatchedCircuitOp::BatchOffsets(batch_data.indices().to_vec()),
        );
    }

    pub fn from_pure_circuit(circuit: Circuit<PureCircuit>, max_target_qubits: usize) -> Self {
        let mut batched_circuit = Self {
            batcher: GateBatcher::new(max_target_qubits),
            instruction_lookup: BTreeMap::new(),
            data: GateBatchData::new(),
        };

        for gate in circuit.instructions() {
            batched_circuit.batcher.add_gate(gate);
        }

        batched_circuit.flush_batches(0);

        batched_circuit
    }

    pub fn from_hybrid_circuit(
        circuit: Circuit<HybridCircuit>,
        max_targets_per_batch: usize,
    ) -> Self {
        let mut batched_circuit = Self {
            batcher: GateBatcher::new(max_targets_per_batch),
            instruction_lookup: BTreeMap::new(),
            data: GateBatchData::new(),
        };

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


}

impl From<Circuit<PureCircuit>> for BatchedCircuit {
    fn from(value: Circuit<PureCircuit>) -> Self {
        BatchedCircuit::from_pure_circuit(value, BatchedCircuit::DEFAULT_MAX_TARGET_QUBITS)
    }
}

impl From<Circuit<HybridCircuit>> for BatchedCircuit {
    fn from(value: Circuit<HybridCircuit>) -> Self {
        BatchedCircuit::from_hybrid_circuit(value, BatchedCircuit::DEFAULT_MAX_TARGET_QUBITS)
    }
}
