use std::{collections::HashMap, mem};

use nalgebra::{Complex, Matrix2};

use crate::{
    ext::get_gate2_matrix,
    gate::{Gate, GateType, QBits},
};

type NodeId = usize;
type BatchId = usize;

#[derive(Debug, Clone)]
struct GateNode {
    batch_id: BatchId,
    matrix: Matrix2<Complex<f32>>,
    target: QBits,
    control: QBits,
    n_gates: usize,
}

impl GateNode {
    fn new(batchid: BatchId, init: &Gate) -> Self {
        Self {
            batch_id: batchid,
            matrix: matrix2(init),
            target: init.get_target_bits(),
            control: init.get_control_bits(),
            n_gates: 1,
        }
    }

    fn mul(&mut self, gate: &Gate) {
        self.matrix *= matrix2(gate);
        self.n_gates += 1;
    }
}

// Matrix helper
fn matrix2(gate2: &Gate) -> Matrix2<Complex<f32>> {
    get_gate2_matrix(gate2).expect("gate size mismatch")
}

#[derive(Debug, Clone)]
struct GateBatch {
    id: BatchId,
    nodes: Vec<NodeId>,
    target_union: QBits,
    retired: bool,
}

impl GateBatch {
    fn new(id: BatchId) -> Self {
        Self {
            id,
            nodes: Vec::new(),
            target_union: QBits::default(),
            retired: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchCommand {
    pub start_index: u32,
    pub size: u32,
    pub targets: QBits,
}

#[derive(Debug, Clone)]
pub struct BatchedData {
    gate_data: Vec<Complex<f32>>,
    target_data: Vec<u32>,
    control_data: Vec<u32>,
    len: usize,
}

impl BatchedData {
    pub fn new() -> Self {
        Self {
            gate_data: Vec::new(),
            target_data: Vec::new(),
            control_data: Vec::new(),
            len: 0,
        }
    }

    pub fn with_capacity(nodes: usize) -> Self {
        Self {
            gate_data: Vec::with_capacity(nodes * 4),
            target_data: Vec::with_capacity(nodes),
            control_data: Vec::with_capacity(nodes),
            len: 0,
        }
    }

    pub fn append(&mut self, mut other: BatchedData) {
        self.gate_data.append(&mut other.gate_data);
        self.target_data.append(&mut other.target_data);
        self.control_data.append(&mut other.control_data);
        self.len += other.len;
    }

    pub fn gate_data(&self) -> &[Complex<f32>] {
        &self.gate_data
    }

    pub fn target_data(&self) -> &[u32] {
        &self.target_data
    }

    pub fn control_data(&self) -> &[u32] {
        &self.control_data
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn insert_from(&mut self, gate_batch: &GateBatch, gate_nodes: &[GateNode]) {
        let mut node_ids = gate_batch.nodes.clone();
        node_ids.sort_unstable();

        for node_id in node_ids {
            let gate_node = &gate_nodes[node_id];

            self.gate_data
                .extend_from_slice(gate_node.matrix.as_slice());
            self.target_data
                .push(gate_node.target.get_bitstring() as u32);
            self.control_data
                .push(gate_node.control.get_bitstring() as u32);
            self.len += 1;
        }
    }
}

#[derive(Debug, Clone)]
pub struct GateBatcher {
    // qubit -> NodeId  (the frontier node for that qubit)
    frontier: HashMap<usize, NodeId>,

    // BatchId -> GateBatch
    batches: Vec<GateBatch>,

    // NodeId -> GateNode
    nodes: Vec<GateNode>,

    max_target_qubits: usize,

    data: BatchedData,
}

impl GateBatcher {
    pub fn new(max_target_qubits: usize) -> Self {
        Self {
            frontier: HashMap::new(),
            batches: Vec::new(),
            nodes: Vec::new(),
            max_target_qubits,
            data: BatchedData::new(),
        }
    }

    pub fn add_gate(&mut self, gate: &Gate) {
        match gate.get_type() {
            GateType::SWAP => self.add_swap(gate),
            _ => self.add_gate2(gate),
        }
    }

    /// Clears frontier. Subsequent gate additions will be added to new batches.
    ///
    /// Returns commands for all flushed batches as a `Vec<BatchCommand>`
    pub fn flush_batches(&mut self) -> Vec<BatchCommand> {
        let _ = mem::take(&mut self.frontier);
        let batches = mem::take(&mut self.batches);
        let nodes = mem::take(&mut self.nodes);

        let mut batch_ids: Vec<BatchId> = batches
            .iter()
            .enumerate()
            .filter_map(|(batch_id, batch)| {
                if batch.retired || batch.nodes.is_empty() {
                    None
                } else {
                    Some(batch_id)
                }
            })
            .collect();

        batch_ids.sort_by_key(|&batch_id| {
            batches[batch_id]
                .nodes
                .iter()
                .copied()
                .min()
                .unwrap_or(usize::MAX)
        });

        let mut batch_commands = Vec::<BatchCommand>::new();

        for batch_id in batch_ids {
            let batch = &batches[batch_id];

            batch_commands.push(BatchCommand {
                start_index: self.data.len() as u32,
                size: batch.nodes.len() as u32,
                targets: batch.target_union,
            });

            self.data.insert_from(batch, &nodes);
        }

        batch_commands
    }

    pub fn data(&self) -> &BatchedData {
        &self.data
    }

    fn add_swap(&mut self, gate: &Gate) {
        // This function only handles swap
        assert_eq!(gate.get_type(), GateType::SWAP, "expected SWAP gate");

        let mut controls = gate.get_controls();
        let targets = gate.get_targets();
        let (t0, t1) = (targets[0], targets[1]);

        controls.push(t1);

        let g0 = Gate::new(GateType::X, &[t0], &[t1]).unwrap();
        let g1 = Gate::new(GateType::X, &controls, &[t0]).unwrap();

        self.add_gate2(&g0);
        self.add_gate2(&g1);
        self.add_gate2(&g0);
    }

    /* Adding a gate
     *
     * 1. Collect frontier NodeId for all qubits (target | control) in gate.
     *    - If all map to the same NodeId AND signatures match:
     *      multiply matrix in-place, done.
     *    - If all map to the same NodeId BUT signatures differ:
     *      create new node in same batch, update frontier.
     *    - If they differ:
     *
     * 2. Check the combined target qubit range if you were to combine the nodes
     *    - If exceeds limit:
     *      create new batch, new node in it, update frontier for gate's qubits (targets and controls).
     *      (old batches untouched, still accessible if future gates point back to them)
     *    - If within limit:
     *
     * 3. Merge all involved batches into one (lower id absorbs).
     *    Create new node in merged batch, update frontier.
     */
    fn add_gate2(&mut self, gate: &Gate) {
        // This function only handles single qubit gates
        assert_eq!(gate.get_type().arity(), 1, "expected single qubit gates");

        let gate_target = gate.get_target_bits();
        let gate_control = gate.get_control_bits();
        let touched_qubits = gate_target.union(gate_control).get_indices();

        let mut frontier_nodes: HashMap<NodeId, BatchId> = HashMap::new();
        for &q in &touched_qubits {
            if let Some(&node_id) = self.frontier.get(&q) {
                frontier_nodes.insert(node_id, self.nodes[node_id].batch_id);
            }
        }

        let mut combined_targets = gate_target;
        for (_, &batch_id) in &frontier_nodes {
            combined_targets = combined_targets.union(self.batches[batch_id].target_union);
        }

        // New batch and node if touched qubits lack frontier
        // Or if combined targets exceeds specified maximum
        if frontier_nodes.is_empty() || combined_targets.count() > self.max_target_qubits {
            let new_batch_id = self.new_batch();
            let new_node_id = self.new_node(new_batch_id, gate);
            self.update_frontier(&touched_qubits, new_node_id);

            return;
        }

        // If all discovered frontier entries map to the same NodeId
        //
        // Note that some touched qubits may still have no frontier entry at all.
        // This is okay: if the signature differs we create a new node below and
        // update the frontier for every touched qubit, including the previously
        // untouched ones
        if frontier_nodes.len() == 1 {
            let (&node_id, &batch_id) = frontier_nodes.iter().next().unwrap();

            let pred_node = &self.nodes[node_id];

            let siganture_match =
                pred_node.control == gate_control && pred_node.target == gate_target;

            // Signature match means the full qubit footprint is identical
            // (same targets and same controls). In that case the frontier for
            // all touched qubits must already point at this node from when it
            // was first created, so no frontier update is needed here
            if siganture_match {
                self.nodes[node_id].mul(gate);

                return;
            }
            // Otherwise we append a new node to the same batch and make it the
            // new frontier for every touched qubit. This also fills in frontier
            // entries for touched qubits that previously had none
            else {
                let new_node_id = self.new_node(batch_id, gate);
                self.update_frontier(&touched_qubits, new_node_id);

                return;
            }
        }

        // Collect BatchIds and merge batches
        let mut batch_ids: Vec<BatchId> = frontier_nodes.values().copied().collect();
        batch_ids.sort_unstable();
        batch_ids.dedup();

        let merged_batch = self.merge_batches(&batch_ids);
        let new_node_id = self.new_node(merged_batch, gate);
        self.update_frontier(&touched_qubits, new_node_id);
    }

    fn new_batch(&mut self) -> BatchId {
        let id = self.batches.len();
        self.batches.push(GateBatch::new(id));
        id
    }

    fn new_node(&mut self, batch_id: BatchId, gate: &Gate) -> NodeId {
        let id = self.nodes.len();
        let batch = &mut self.batches[batch_id];

        self.nodes.push(GateNode::new(batch_id, gate));
        batch.nodes.push(id);
        batch.target_union = batch.target_union.union(gate.get_target_bits());

        id
    }

    fn update_frontier(&mut self, qubits: &[usize], to: NodeId) {
        for &q in qubits {
            self.frontier.insert(q, to);
        }
    }

    fn merge_batches(&mut self, batches: &[BatchId]) -> BatchId {
        let merged_batchid = self.new_batch();

        let (old_batches, merged_tail) = self.batches.split_at_mut(merged_batchid);
        let merged_batch = &mut merged_tail[0];

        // Add all nodes of batches to new merged batch, and retire old batches
        for &batchid in batches {
            let old_batch = &mut old_batches[batchid];

            merged_batch.nodes.append(&mut old_batch.nodes);
            merged_batch.target_union = merged_batch.target_union.union(old_batch.target_union);

            old_batch.retired = true;
        }

        // Update batch_id on all moved nodes
        for &node_id in &merged_batch.nodes {
            self.nodes[node_id].batch_id = merged_batchid;
        }

        merged_batchid
    }
}

#[cfg(test)]
mod tests {
    use crate::{circuit::Circuit, gpu_sv_simulator::gate_batcher::GateBatcher};

    #[test]
    fn test_example_circuit() {
        let circ = Circuit::new(3)
            .x(0)
            .x(2)
            .y(0)
            .cx(&[1], 0)
            .swap(0, 2)
            .cy(&[0], 2);

        let mut circ_ir = GateBatcher::new(2);

        for gate in circ.instructions() {
            match &gate {
                crate::instruction::PureInstruction::Gate(gate) => circ_ir.add_gate(gate),
                crate::instruction::PureInstruction::Call(_, _, _) => unreachable!(),
            }
        }

        assert!(circ_ir.nodes.len() == 6);
        assert_eq!(circ_ir.batches[0].retired, true);
        assert_eq!(circ_ir.batches[1].retired, true);
        assert_eq!(circ_ir.batches[2].retired, false);
        assert_eq!(circ_ir.batches[2].nodes, vec![0, 2, 1, 3, 4, 5]);
        assert_eq!(circ_ir.frontier.get(&1), Some(&2));
    }
}
