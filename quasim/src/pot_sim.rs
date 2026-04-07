use std::{collections::BTreeMap, ops::Index, usize};

use nalgebra::{Complex, DVector, Matrix2};

use crate::{
    cart,
    circuit::{Circuit, CircuitBehaviour, HybridCircuit, pc::CircuitPc},
    expr_dsl::{BitExpr, BoolExpr},
    ext::{BitMaskIter, TargetIter},
    gate::{Gate, GateType, QBits},
    instruction::Instruction,
    register_file::RegisterFile,
    simulator::{DebuggableSimulator, HybridSimulator, StoredCircuitSimulator},
};

pub struct GenericSim<C: StateCollection> {
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    state: C,
    register_file: RegisterFile,
    /// dvector_state, remove for system change
    null_state: DVector<Complex<f64>>,
}

impl<C: StateCollection> GenericSim<C> {
    pub fn collapse_peek(&self) -> usize {
        let mut ri = rand::random_range(0.0..1.0);
        for state in 0..self.circuit.n_qubits() {
            let val = self.state.state(state.into());
            let prob = val.norm();
            ri -= prob;
            if ri <= 0.0 {
                return state;
            }
        }
        !(usize::MAX << self.circuit.n_qubits())
    }

    fn handle_gate(&mut self, gate: &Gate) {
        self.pc.increment();
        let ctrl = *gate.get_control_bits();
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
                    let dont_care =
                        !(ctrl | (1 << target)) & !(usize::MAX << self.circuit.n_qubits());
                    let combinations = BitMaskIter::from(dont_care);
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

                let dont_care =
                    !(ctrl | t1_mask | t2_mask) & !(usize::MAX << self.circuit.n_qubits());
                let combinations = BitMaskIter::from(dont_care);
                for combination in combinations {
                    let qs1 = ctrl | combination | t1_mask;
                    let qs2 = ctrl | combination | t2_mask;
                    let s1 = self.state.state(qs1);
                    let s2 = self.state.insert(qs2, s1);
                    self.state.insert(qs1, s2);
                }
            }
        }
    }

    fn handle_measure_bit(&mut self, target: usize, (reg, c_target): (&str, usize)) {
        self.pc.increment();
        let collapsed = self.collapse_peek();
        let q_mask = 1 << target;
        let c_mask = 1 << c_target;
        let q_masked = collapsed & q_mask;
        let c_masked = (q_masked >> target) << c_target;

        let mut val = self.register_file[reg].read();
        val &= !c_mask;
        val |= c_masked;
        self.register_file[reg].write(val);

        self.state.retain_norm(|state, _| {
            let s_masked = state & q_mask;
            s_masked ^ q_masked == 0
        });
    }

    fn handle_measure_all(&mut self, reg: &str) {
        self.pc.increment();
        let collapsed = self.collapse_peek();
        self.register_file[reg].write(collapsed);
        self.state.retain(|_, _| false);
        self.state.insert(collapsed.into(), cart!(1));
    }

    fn handle_jump(&mut self, pc: usize) {
        self.pc.jump(pc);
    }

    fn handle_jump_if(&mut self, expr: &BoolExpr, pc: usize) {
        match expr.eval(&self.register_file) {
            true => self.handle_jump(pc),
            false => self.pc.increment(),
        }
    }

    fn handle_assign(&mut self, expr: &BitExpr, reg: &str) {
        self.pc.increment();
        let val = expr.eval(&self.register_file);
        self.register_file[reg].write(val);
    }

    fn handle_call(&mut self, name: String, lsq: usize, ctrl: QBits) {
        self.pc.jump_and_link(name, lsq, ctrl);
    }
}

impl<C: StateCollection> DebuggableSimulator for GenericSim<C> {
    type Storage = C;
    type State = Complex<f64>;

    fn next(&mut self) -> bool {
        let Some(inst) = self.circuit.instruction(&self.pc) else {
            // End of (sub) circuit: Try to return
            if self.pc.ret() {
                return true;
            }

            // Could not return: End of circuit
            return false;
        };

        match inst {
            Instruction::Gate(gate) => self.handle_gate(&gate),
            Instruction::MeasureBit(target, (reg, c_target)) => {
                self.handle_measure_bit(target, (&reg, c_target))
            }
            Instruction::MeasureAll(reg) => self.handle_measure_all(&reg),
            Instruction::Jump(pc) => self.handle_jump(pc),
            Instruction::JumpIf(expr, pc) => self.handle_jump_if(&expr, pc),
            Instruction::Assign(expr, reg) => self.handle_assign(&expr, &reg),
            Instruction::Call(name, lsq, ctrl) => self.handle_call(name, lsq, ctrl),
        }

        for i in 0..(1 << self.circuit.n_qubits()) {
            self.null_state[i] = self.state.state(i.into());
        }

        true
    }

    fn double_ended(&self) -> bool {
        false
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<crate::instruction::Instruction>) {
        (&self.pc, self.circuit.instruction(&self.pc))
    }

    fn current_state(&self) -> &C {
        &self.state
    }

    fn collapse_peek(&self) -> usize {
        todo!()
    }
}

impl<C: StateCollection> StoredCircuitSimulator for GenericSim<C> {
    type B = HybridCircuit;

    fn circuit(&self) -> &Circuit<Self::B> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<Self::B> {
        &mut self.circuit
    }
}

impl<C: StateCollection> HybridSimulator for GenericSim<C> {
    fn registers(&self) -> &RegisterFile {
        &self.register_file
    }
}

impl<B, C> TryFrom<Circuit<B>> for GenericSim<C>
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
    C: StateCollection,
{
    type Error = GenericSimError;

    fn try_from(value: Circuit<B>) -> Result<Self, Self::Error> {
        let mut state = DVector::zeros(1 << value.n_qubits());
        state[0] = cart!(1);
        Ok(Self {
            state: C::new(value.n_qubits()),
            register_file: RegisterFile::from(value.registers()),
            circuit: value.into(),
            pc: CircuitPc::new(0),
            null_state: state,
        })
    }
}

pub trait StateCollection {
    fn new(n_qbits: usize) -> Self;
    fn state(&self, qbits: usize) -> Complex<f64>;
    fn insert(&mut self, qbits: usize, state: Complex<f64>) -> Complex<f64>;
    fn non_zero(&self) -> impl Iterator<Item = (usize, Complex<f64>)>;
    fn normalize_with(&mut self, divisor: f64);
    fn retain<F: FnMut(usize, Complex<f64>) -> bool>(&mut self, f: F);
    fn retain_norm<F: FnMut(usize, Complex<f64>) -> bool>(&mut self, f: F) {
        let mut total_prob = 0.0;
        let mut f = f;
        self.retain(|state, prob| {
            if f(state, prob) {
                total_prob += prob.norm_sqr();
                true
            } else {
                false
            }
        });

        self.normalize_with(total_prob);
    }
    fn pair_mut(&mut self, qbits: usize, target: usize) -> StatePairMut<'_, Self> {
        StatePairMut {
            collection: self,
            qbits,
            target_mask: 1 << target,
        }
    }
}

impl StateCollection for DVector<Complex<f64>> {
    fn new(n_qbits: usize) -> Self {
        let mut d = DVector::zeros(1 << n_qbits);
        d[0] = cart!(1);
        d
    }

    fn state(&self, qbits: usize) -> Complex<f64> {
        self[qbits]
    }

    fn insert(&mut self, qbits: usize, state: Complex<f64>) -> Complex<f64> {
        let ret = self[qbits];
        self[qbits] = state;
        ret
    }

    fn non_zero(&self) -> impl Iterator<Item = (usize, Complex<f64>)> {
        self.iter().enumerate().filter_map(|(i, val)| {
            if val.norm() > 0.0 {
                Some((i, *val))
            } else {
                None
            }
        })
    }

    fn normalize_with(&mut self, divisor: f64) {
        for i in 0..self.len() {
            self[i] /= divisor;
        }
    }

    fn retain<F: FnMut(usize, Complex<f64>) -> bool>(&mut self, f: F) {
        let mut f = f;
        for i in 0..self.len() {
            let qbits = i.into();
            if !f(qbits, self.state(qbits)) {
                self.insert(qbits, cart!(0));
            }
        }
    }
}

pub struct StateMaybe<C: IncompleteStateCollection, const ZM: usize = 1000000> {
    collection: C,
}

impl<const ZM: usize, C> StateMaybe<C, ZM>
where
    C: IncompleteStateCollection,
{
    const ZERO_MARGIN: f64 = 1.0 / ZM as f64;
}

impl<const ZM: usize, C> StateCollection for StateMaybe<C, ZM>
where
    C: IncompleteStateCollection,
{
    fn new(_: usize) -> Self {
        Self {
            collection: C::new(),
        }
    }

    fn state(&self, qbits: usize) -> Complex<f64> {
        self.collection.state(qbits).unwrap_or(cart!(0))
    }

    fn insert(&mut self, qbits: usize, state: Complex<f64>) -> Complex<f64> {
        if state.norm() < Self::ZERO_MARGIN {
            self.collection.remove(qbits).unwrap_or(cart!(0))
        } else {
            self.collection.insert(qbits, state).unwrap_or(cart!(0))
        }
    }

    fn non_zero(&self) -> impl Iterator<Item = (usize, Complex<f64>)> {
        self.collection.iter()
    }

    fn retain<F: FnMut(usize, Complex<f64>) -> bool>(&mut self, f: F) {
        self.collection.retain(f);
    }

    fn normalize_with(&mut self, divisor: f64) {
        self.collection.normalize_with(divisor);
    }
}

impl<S: IncompleteStateCollection> Index<usize> for StateMaybe<S> {
    type Output = Complex<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        self.collection.state_ref(index).unwrap_or(&cart!(0))
    }
}

pub trait IncompleteStateCollection {
    fn new() -> Self;
    fn state(&self, qbits: usize) -> Option<Complex<f64>>;
    fn state_ref(&self, qbits: usize) -> Option<&Complex<f64>>;
    fn insert(&mut self, qbits: usize, state: Complex<f64>) -> Option<Complex<f64>>;
    fn remove(&mut self, qbits: usize) -> Option<Complex<f64>>;
    fn normalize_with(&mut self, divisor: f64);
    fn iter(&self) -> impl Iterator<Item = (usize, Complex<f64>)>;
    fn retain<F: FnMut(usize, Complex<f64>) -> bool>(&mut self, f: F);
}

impl IncompleteStateCollection for BTreeMap<usize, Complex<f64>> {
    fn new() -> Self {
        let mut s = Self::new();
        s.insert(0, cart!(1));
        s
    }

    fn state(&self, qbits: usize) -> Option<Complex<f64>> {
        self.get(&qbits).copied()
    }

    fn state_ref(&self, qbits: usize) -> Option<&Complex<f64>> {
        self.get(&qbits)
    }

    fn insert(&mut self, qbits: usize, state: Complex<f64>) -> Option<Complex<f64>> {
        self.insert(qbits, state)
    }

    fn remove(&mut self, qbits: usize) -> Option<Complex<f64>> {
        self.remove(&qbits)
    }

    fn iter(&self) -> impl Iterator<Item = (usize, Complex<f64>)> {
        self.keys().copied().zip(self.values().copied())
    }

    fn retain<F: FnMut(usize, Complex<f64>) -> bool>(&mut self, f: F) {
        let mut f = f;
        self.retain(|k, v| f(*k, *v));
    }

    fn normalize_with(&mut self, divisor: f64) {
        for val in self.values_mut() {
            *val /= divisor
        }
    }
}

#[derive(Debug)]
pub struct StatePairMut<'a, C: StateCollection + ?Sized> {
    collection: &'a mut C,
    qbits: usize,
    target_mask: usize,
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

#[derive(Debug, thiserror::Error)]
pub enum GenericSimError {}

#[cfg(test)]
mod tests {
    use super::{GenericSim, StateMaybe};
    use crate::common_test;
    use nalgebra::{Complex, DVector};
    use std::collections::BTreeMap;

    macro_rules! expand_gen {
        (dvector) => {
            DVector<Complex<f64>>
        };
        (btree) => {
            StateMaybe<BTreeMap<usize, Complex<f64>>>
        }
    }
    macro_rules! def_test {
        ($tst:ident; $($id:ident, $ty:ident);+ $(;)?) => {
            $(
                #[test]
                fn $id() {
                    common_test::$tst::<GenericSim<expand_gen!($ty)>>();
                }
            )+
        };
    }

    def_test!(
        apply_gates;
        apply_gates_dvector, dvector;
        apply_gates_btree, btree;
    );

    def_test!(
        double_sub;
        double_sub_dvector, dvector;
        double_sub_btree, btree;
    );

    def_test!(
        deep_sub;
        deep_sub_dvector, dvector;
        deep_sub_btree, btree;
    );

    def_test!(
        deep_ctrl_sub;
        deep_ctrl_sub_dvector, dvector;
        deep_ctrl_sub_btree, btree;
    );

    def_test!(
        hybrid_test;
        hybrid_test_dvector, dvector;
        hybrid_test_btree, btree;
    );

    def_test!(
        register_test;
        register_test_dvector, dvector;
        register_test_btree, btree;
    );

    def_test!(
        test_measure_overwrites_with_zero;
        test_measure_overwrites_with_zero_dvector, dvector;
        test_measure_overwrites_with_zero_btree, btree;
    );
}
