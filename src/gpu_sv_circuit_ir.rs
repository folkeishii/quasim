use std::{collections::{BTreeMap, HashMap}, hash::Hash};

use crate::{circuit::{Circuit, HybridCircuit, PureCircuit}, gate_batcher::{BatchHandle, GateBatchData, GateBatcher}, instruction::Instruction};



struct GPUSVCircuitIR {
    batcher: GateBatcher,
    instruction_lookup: BTreeMap<usize, IRInstruction>,
    batch_data: GateBatchData,
}

enum IRInstruction {
    Batch(usize),
    Inst(Instruction)
}

impl GPUSVCircuitIR {
    pub fn from_pure_circuit(circuit: Circuit<PureCircuit>, max_targets_per_batch: usize) -> Self {
        let mut batcher = GateBatcher::new(max_targets_per_batch);
        
        for gate in circuit.instructions() {
            batcher.add_gate(gate);
        }

        let mut instruction_lookup = BTreeMap::new();
        instruction_lookup.insert(0, IRInstruction::Batch(0));
        let batch_data = batcher.close_batches();

        Self {
            batcher, instruction_lookup, batch_data,
        }
    }

    pub fn from_hybrid_circuit(
        circuit: Circuit<HybridCircuit>,
        max_targets_per_batch: usize,
    ) -> Self {
        let mut batcher = GateBatcher::new(max_targets_per_batch);
        let mut instruction_lookup = BTreeMap::new();
        let mut batch_data = GateBatchData::new();

        let mut batch_inst_index = 0;

        for (inst_index, inst) in circuit.instructions().iter().enumerate() {
            match inst {
                Instruction::Gate(gate) => {
                    batcher.add_gate(gate);
                },
                _ => {
                    let batch = batcher.close_batches();
                    let batch_inst = IRInstruction::Batch(batch_data.len());
                    let other_inst = IRInstruction::Inst(*inst);

                    batch_data.append(batch);
                    


                    instruction_lookup.insert(batch_inst_index, batch_inst);
                    instruction_lookup.insert(inst_index, other_inst);

                    batch_inst_index = inst_index + 1;
                }
            }
        }

        GPUSVCircuitIR {
            batcher, instruction_lookup, batch_data
        }
    }
}

impl From<Circuit<PureCircuit>> for GateBatcher {
    fn from(value: Circuit<PureCircuit>) -> Self {
        // Choose some default max target qubit value
        GateBatcher::from_pure_circuit(value, 3)
    }
}