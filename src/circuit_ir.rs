use std::collections::{HashMap};

use nalgebra::{Complex, Matrix2};

use crate::{
    circuit::{Circuit, PureCircuit},
    ext::get_gate2_matrix,
    gate::{Gate, GateType, QBits},
};

type NodeId = usize;
type BatchId = usize;

#[derive(Debug)]
struct GateNode {
    // Not sure if we will need these ids directly on this struct but we will see...
    // id: NodeId,
    batch_id: BatchId,
    matrix: Matrix2<Complex<f64>>,
    target: QBits,
    control: QBits,
    n_gates: usize,
    // prev: Vec<NodeId>,
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
fn matrix2(gate2: &Gate) -> Matrix2<Complex<f64>> {
    get_gate2_matrix(gate2).expect("gate size mismatch")
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct CircuitIR {
    // qubit -> NodeId  (the frontier node for that qubit)
    frontier: HashMap<usize, NodeId>,

    // BatchId -> GateBatch
    batches: Vec<GateBatch>,

    // NodeId -> GateNode
    nodes: Vec<GateNode>,

    max_target_qubits: usize,
}

impl CircuitIR {
    pub fn new(max_targets_per_batch: usize) -> Self {
        Self {
            frontier: HashMap::new(),
            batches: Vec::new(),
            nodes: Vec::new(),
            max_target_qubits: max_targets_per_batch,
        }
    }

    pub fn from_pure_circuit(circuit: Circuit<PureCircuit>, max_targets_per_batch: usize) -> Self {
        let mut circuit_ir = CircuitIR::new(max_targets_per_batch);

        for gate in circuit.instructions() {
            // Special case for SWAP
            // represent swap as three CNOTs
            if gate.get_type() == GateType::SWAP {
                let mut controls = gate.get_controls();
                let targets = gate.get_targets();
                let t0 = targets[0];
                let t1 = targets[1];

                controls.push(t1);

                let g0 = Gate::new(GateType::X, &[t0], &[t1]).unwrap();
                let g1 = Gate::new(GateType::X, &controls, &[t0]).unwrap();

                circuit_ir.add_gate(&g0);
                circuit_ir.add_gate(&g1);
                circuit_ir.add_gate(&g0);
            } else {
                circuit_ir.add_gate(gate);
            }
        }

        circuit_ir
    }

    /* Adding a gate
     *
     * 1. Collect frontier NodeId for all qubits (target | control) in gate.
     *    - If all map to the same NodeId AND signatures match:
     *      multiply matrix in-place, done.
     *    - If all map to the same NodeId BUT signatures differ:
     *      create new node in same batch, prev = [that node], update frontier.
     *    - If they differ (multiple nodes, or some qubits have no frontier):
     *
     * 2. Check the combined target qubit range if you were to combine the nodes
     *    - If exceeds limit:
     *      create new batch, new node in it, update frontier for gate's qubits (targets and controls).
     *      (old batches untouched, still accessible if future gates point back to them)
     *    - If within limit:
     *
     * 3. Merge all involved batches into one (lower id absorbs).
     *    Create new node in merged batch, prev = [all distinct frontier nodes], update frontier.
     */
    pub fn add_gate(&mut self, gate: &Gate) {
        let gate_target = gate.get_target_bits();
        let gate_control = gate.get_control_bits();
        let gate_qubits = gate_target.union(gate_control);

        let touched_qubits = gate_qubits.get_indices();

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

/** Getting batch matrix data
 *
 * Go through the nodes vec for each batch in order and create a contigous vec of
 * matrix data, target data, control data, which will be passed to the gpu once on simulator init.
 * Then i will pass a command specifying offset in matrix and target/control data and size of batch
 * in order to execute the batch on the gpu.
 *
 * Since nodes are always appended after their predecessors during construction,
 * iterating the vec in order is a valid execution sequence.
 */

impl From<Circuit<PureCircuit>> for CircuitIR {
    fn from(value: Circuit<PureCircuit>) -> Self {
        // Choose some default max target qubit value
        CircuitIR::from_pure_circuit(value, 3)
    }
}

mod tests {
    use crate::{circuit::Circuit, circuit_ir::CircuitIR};

    #[test]
    fn example_test_circuit() {
        let circ = Circuit::new(3)
            .x(0)
            .x(2)
            .y(0)
            .cx(&[1], 0)
            .swap(0, 2)
            .cy(&[0], 2);

        let circ_ir = CircuitIR::from_pure_circuit(circ, 2);

        assert!(circ_ir.nodes.len() == 6);
        assert_eq!(circ_ir.batches[0].retired, true);
        assert_eq!(circ_ir.batches[1].retired, true);
        assert_eq!(circ_ir.batches[2].retired, false);
        assert_eq!(circ_ir.batches[2].nodes, vec![0, 2, 1, 3, 4, 5]);
        assert_eq!(circ_ir.frontier.get(&1), Some(&2));
    }
}