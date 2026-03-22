use std::collections::HashMap;

use nalgebra::{Complex, Matrix2};

use crate::{
    circuit::{Circuit, HybridCircuit},
    ext::{get_gate2_data, get_gate2_matrix},
    gate::{Gate, GateType, QBits},
};

type NodeId = usize;
type BatchId = usize;

struct GateNode {
    // Not sure if we will need these ids directly on this struct but we will see...
    // id: NodeId,
    // batch_id: BatchId,

    matrix: Matrix2<Complex<f64>>,
    target: QBits,
    control: QBits,
    n_gates: usize,

    // prev: Vec<NodeId>,
}

impl GateNode {
    fn new(init: &Gate) -> Self {
        Self {
            matrix: matrix2(init),
            target: init.get_target_bits(),
            control: init.get_control_bits(),
            n_gates: 1,
        }
    }

    // TODO change this, this doesnt really match that well with new algo description
    fn add(mut self, gate: &Gate) -> GateNode {
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

            self.add(&g0).add(&g1).add(&g0)
        }
        // If gate signature match we can just multiply matrices
        else if self.target == gate.get_target_bits() && self.control == gate.get_control_bits() {
            self.matrix *= matrix2(gate);

            self
        }
        // If gate signature does not match we add it as a new node.
        else {
            // GateNode::new(gate).merge_with(vec![self])
            self // temp to avoid error
        }
    }
}

// Matrix helper
fn matrix2(gate2: &Gate) -> Matrix2<Complex<f64>> {
    get_gate2_matrix(gate2).expect("gate size mismatch")
}

struct GateBatch {
    id: BatchId,
    nodes: Vec<NodeId>,
    target_union: QBits,
}

struct CircuitIR {
    // qubit -> (BatchId, NodeId)  (the frontier node for that qubit, along with associated batch)
    frontier: HashMap<usize, (BatchId, NodeId)>,

    // BatchId -> GateBatch
    batches: Vec<GateBatch>,

    // NodeId -> GateNode
    nodes: Vec<GateNode>,
}

/** Adding a gate
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

impl CircuitIR {
    
}

impl From<Circuit<HybridCircuit>> for CircuitIR {
    fn from(value: Circuit<HybridCircuit>) -> Self {
        todo!()
    }
}
