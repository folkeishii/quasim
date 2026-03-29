use std::{collections::BTreeMap, ops::Mul};

use nalgebra::{Complex, DVector, Matrix2, SMatrix, dvector};

use crate::{
    cart,
    circuit::{Circuit, CircuitBehaviour, HybridCircuit, pc::CircuitPc},
    ext::{BitMaskIter, TargetIter},
    gate::{GateType, QBits},
    instruction::Instruction,
    simulator::DebuggableSimulator,
};

pub struct GenericSim<C: StateCollection> {
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    state: C,
    /// dvector_state, remove for system change
    null_state: DVector<Complex<f64>>,
}

impl<C: StateCollection> DebuggableSimulator for GenericSim<C> {
    fn next(&mut self) -> Option<&DVector<Complex<f64>>> {
        let Some(inst) = self.circuit.instruction(&self.pc) else {
            // End of (sub) circuit: Try to return
            if self.pc.ret() {
                return Some(&self.null_state);
            }

            // Could not return: End of circuit
            return None;
        };

        match inst {
            Instruction::Gate(gate) => {
                self.pc.increment();
                let ctrl = gate.get_control_bits();
                let mut targets = TargetIter::from(gate.get_target_bits());
                match gate.get_type() {
                    ty @ GateType::X
                    | ty @ GateType::Y
                    | ty @ GateType::Z
                    | ty @ GateType::H
                    | ty @ GateType::U(_, _, _)
                    | ty @ GateType::S => {
                        let mat = ty.unchecked_matrix2x2();
                        for target in targets {
                            let dont_care = !(*ctrl | (1 << target));
                            let combinations = BitMaskIter::from(dont_care).map(Into::into);
                            for combination in combinations {
                                let base = ctrl | combination;
                                let mut pair = self.state.pair_mut(base, target);
                                pair.apply2x2(mat);
                            }
                        }
                    }
                    GateType::SWAP => {
                        let t1 = targets.next().expect("invalid circuit");
                        let t1_mask = 1 << t1;
                        let t2 = targets.next().expect("invalid circuit");
                        let t2_mask = 1 << t2;

                        let dont_care = !(*ctrl | t1_mask | t2_mask);
                        let combinations = BitMaskIter::from(dont_care).map(Into::into);
                        for combination in combinations {
                            let qs1 = ctrl | combination | t1_mask.into();
                            let qs2 = ctrl | combination | t2_mask.into();
                            let s1 = self.state.state(qs1);
                            let s2 = self.state.insert(qs2, s1);
                            self.state.insert(qs1, s2);
                        }
                    }
                }
            }
            Instruction::MeasureBit(_, _) => todo!(),
            Instruction::MeasureAll(_) => todo!(),
            Instruction::Jump(_) => todo!(),
            Instruction::JumpIf(expr, _) => todo!(),
            Instruction::Assign(expr, _) => todo!(),
            Instruction::Call(_, _, qbits) => todo!(),
        }

        Some(&self.null_state)
    }

    fn double_ended(&self) -> bool {
        todo!()
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<crate::instruction::Instruction>) {
        todo!()
    }

    fn current_state(&self) -> &DVector<Complex<f64>> {
        todo!()
    }
}

impl<B, C> TryFrom<Circuit<B>> for GenericSim<C>
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
    C: StateCollection,
{
    type Error = ();

    fn try_from(value: Circuit<B>) -> Result<Self, Self::Error> {
        Ok(Self {
            state: C::new(value.n_qubits()),
            circuit: value.into(),
            pc: CircuitPc::new(0),
            null_state: dvector![cart!(0)],
        })
    }
}

pub trait StateCollection {
    fn new(n_qbits: usize) -> Self;
    fn state(&self, qbits: QBits) -> Complex<f64>;
    fn insert(&mut self, qbits: QBits, state: Complex<f64>) -> Complex<f64>;
    fn pair_mut(&mut self, qbits: QBits, target: usize) -> StatePairMut<'_, Self> {
        StatePairMut {
            collection: self,
            qbits,
            target_mask: (1 << target).into(),
        }
    }
}

impl StateCollection for DVector<Complex<f64>> {
    fn new(n_qbits: usize) -> Self {
        let mut d = DVector::zeros(1 << n_qbits);
        d[0] = cart!(1);
        d
    }

    fn state(&self, qbits: QBits) -> Complex<f64> {
        self[*qbits]
    }

    fn insert(&mut self, qbits: QBits, state: Complex<f64>) -> Complex<f64> {
        let ret = self[*qbits];
        self[*qbits] = state;
        ret
    }
}

pub struct StateMaybe<const ZM: usize, C: IncompleteStateCollection> {
    collection: C,
}

impl<const ZM: usize, C> StateMaybe<ZM, C>
where
    C: IncompleteStateCollection,
{
    const ZERO_MARGIN: f64 = 1.0 / ZM as f64;
}

impl<const ZM: usize, C> StateCollection for StateMaybe<ZM, C>
where
    C: IncompleteStateCollection,
{
    fn new(_: usize) -> Self {
        Self {
            collection: C::new(),
        }
    }

    fn state(&self, qbits: QBits) -> Complex<f64> {
        self.collection.state(qbits).unwrap_or(cart!(0))
    }

    fn insert(&mut self, qbits: QBits, state: Complex<f64>) -> Complex<f64> {
        if state.norm() < Self::ZERO_MARGIN {
            self.collection.remove(qbits).unwrap_or(cart!(0))
        } else {
            self.collection.insert(qbits, state).unwrap_or(cart!(0))
        }
    }
}

pub trait IncompleteStateCollection {
    fn new() -> Self;
    fn state(&self, qbits: QBits) -> Option<Complex<f64>>;
    fn insert(&mut self, qbits: QBits, state: Complex<f64>) -> Option<Complex<f64>>;
    fn remove(&mut self, qbits: QBits) -> Option<Complex<f64>>;
}

impl IncompleteStateCollection for BTreeMap<QBits, Complex<f64>> {
    fn new() -> Self {
        Self::new()
    }

    fn state(&self, qbits: QBits) -> Option<Complex<f64>> {
        self.get(&qbits).copied()
    }

    fn insert(&mut self, qbits: QBits, state: Complex<f64>) -> Option<Complex<f64>> {
        self.insert(qbits, state)
    }

    fn remove(&mut self, qbits: QBits) -> Option<Complex<f64>> {
        self.remove(&qbits)
    }
}

#[derive(Debug)]
pub struct StatePairMut<'a, C: StateCollection + ?Sized> {
    collection: &'a mut C,
    qbits: QBits,
    target_mask: QBits,
}

impl<'a, C: StateCollection + ?Sized> StatePairMut<'a, C> {
    pub fn apply2x2(&mut self, gate: Matrix2<Complex<f64>>) {
        let pair = [
            self.collection.state(self.qbits & !self.target_mask),
            self.collection.state(self.qbits | self.target_mask),
        ];

        let res = [
            gate.row(0)[0] * pair[0] + gate.row(0)[1] * pair[1],
            gate.row(1)[0] * pair[0] + gate.row(1)[1] * pair[1],
        ];

        self.collection
            .insert(self.qbits & !self.target_mask, res[0]);
        self.collection
            .insert(self.qbits | self.target_mask, res[1]);
    }
}
