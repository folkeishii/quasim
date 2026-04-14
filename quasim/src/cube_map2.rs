use rayon::prelude::*;
use std::{
    alloc::{Layout, dealloc},
    mem,
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr::NonNull,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use nalgebra::{Complex, Matrix2, Vector2};

use crate::{
    cart,
    circuit::{Circuit, CircuitBehaviour, HybridCircuit, pc::CircuitPc},
    expr_dsl::{BitExpr, BoolExpr},
    ext::{BitSet, TargetIter},
    gate::{Gate, GateType, QBits},
    register_file::RegisterFile,
    simulator::{Debuggable, QuantumState, Sampleable, Simulator, StoredCircuit, StoredRegisters},
};

macro_rules! read {
    ($x:expr) => {
        $x.read().unwrap_or_else(|e| e.into_inner())
    };
}
macro_rules! write {
    ($x:expr) => {
        $x.write().unwrap_or_else(|e| e.into_inner())
    };
}

pub struct CubeSimulator {
    circuit: Circuit<HybridCircuit>,
    pc: CircuitPc,
    state_vector: CubeVector,
    register_file: RegisterFile,
}

impl CubeSimulator {
    fn handle_gate(&mut self, gate: Gate) {
        self.pc.increment();
        self.state_vector.apply_gate(gate);
    }

    fn handle_measure_bit(&mut self, target: usize, (reg, c_target): (&str, usize)) {
        self.pc.increment();
        let collapsed = self.state_vector.collapse();
        let q_mask = 1 << target;
        let c_mask = 1 << c_target;
        let q_masked = collapsed & q_mask;
        let c_masked = (q_masked >> target) << c_target;

        let mut val = self.register_file[reg].read();
        val &= !c_mask;
        val |= c_masked;
        self.register_file[reg].write(val);

        let total_prob = Arc::new(RwLock::new(0.0));
        self.state_vector.map(|state, val| {
            let s_masked = state & q_mask;
            if s_masked ^ q_masked == 0 {
                *write!(total_prob) += val.norm_sqr();
            } else {
                *val = cart!(0);
            }
        });
        let total_prob = read!(total_prob).sqrt();
        self.state_vector.map(|_, val| {
            *val /= total_prob;
        });
    }

    fn handle_measure_all(&mut self, reg: &str) {
        self.pc.increment();
        let collapsed = self.state_vector.collapse();
        self.register_file[reg].write(collapsed);
        self.state_vector.map(|i, val| {
            if i == collapsed {
                *val = cart!(1)
            } else {
                *val = cart!(0)
            }
        });
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

impl Simulator for CubeSimulator {
    type State = CubeVector;
    type BasisValue = Complex<f64>;

    fn run(&mut self) {
        while self.next() {}
    }

    fn reset(&mut self) {
        self.register_file.reset();
        self.pc = Default::default();
        self.state_vector
            .map(|i, v| *v = cart!(1usize.saturating_sub(i)));
    }

    fn state(&self) -> &Self::State {
        &self.state_vector
    }
}

impl Debuggable for CubeSimulator {
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
            crate::instruction::Instruction::Gate(gate) => self.handle_gate(gate),
            crate::instruction::Instruction::MeasureBit(target, (reg, bit)) => {
                self.handle_measure_bit(target, (&reg, bit))
            }
            crate::instruction::Instruction::MeasureAll(reg) => self.handle_measure_all(&reg),
            crate::instruction::Instruction::Jump(pc) => self.handle_jump(pc),
            crate::instruction::Instruction::JumpIf(bool_expr, pc) => {
                self.handle_jump_if(&bool_expr, pc)
            }
            crate::instruction::Instruction::Assign(bit_expr, reg) => {
                self.handle_assign(&bit_expr, &reg)
            }
            crate::instruction::Instruction::Call(sc, lsq, ctrl) => self.handle_call(sc, lsq, ctrl),
        }

        true
    }

    fn double_ended(&self) -> bool {
        false
    }

    fn current_instruction(&self) -> (&CircuitPc, Option<crate::instruction::Instruction>) {
        (&self.pc, self.circuit.instruction(&self.pc))
    }
}

impl<B> TryFrom<Circuit<B>> for CubeSimulator
where
    B: CircuitBehaviour,
    Circuit<B>: Into<Circuit<HybridCircuit>>,
{
    type Error = CubeError;

    fn try_from(value: Circuit<B>) -> Result<Self, Self::Error> {
        Ok(Self {
            state_vector: CubeVector::new(value.n_qubits()),
            register_file: RegisterFile::from(value.registers()),
            circuit: value.into(),
            pc: CircuitPc::default(),
        })
    }
}

impl StoredCircuit for CubeSimulator {
    type B = HybridCircuit;

    fn circuit(&self) -> &Circuit<Self::B> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<Self::B> {
        &mut self.circuit
    }
}

impl StoredRegisters for CubeSimulator {
    fn registers(&self) -> &RegisterFile {
        &self.register_file
    }
}

impl<B: CircuitBehaviour> Sampleable<B> for CubeSimulator where
    Circuit<B>: Into<Circuit<HybridCircuit>>
{
}

#[derive(Debug, Clone)]
pub struct CubeVector {
    zero: Vertex,
}

impl CubeVector {
    pub fn new(n: usize) -> Self {
        let mut zero = Vertex::new(n);
        zero.amplitude = cart!(1);
        Self { zero }
    }

    pub fn apply_gate(&mut self, gate: Gate) {
        let filter: BitSet = BitSet::from(*(gate.get_control_bits()));
        let mut axii = TargetIter::from(gate.get_target_bits());
        let ty = gate.get_type();

        match ty {
            GateType::SWAP => {
                self.zero.filter_swap(
                    axii.next().expect("SWAP expects two target bits"),
                    axii.next().expect("SWAP expects two target bits"),
                    filter,
                    0,
                );
            }
            ty2x2 => {
                while let Some(axis) = axii.next() {
                    self.apply_2x2(axis, filter, ty2x2.unchecked_matrix2x2());
                }
            }
        }
    }

    pub fn for_each<F: Fn(usize, V) + std::marker::Send + std::marker::Sync>(&self, f: F) {
        self.zero.for_each(0.into(), 0, 0, &f);
    }

    pub fn map<F: Fn(usize, &mut V) + std::marker::Send + std::marker::Sync>(&mut self, f: F) {
        self.zero.map(0.into(), 0, 0, &f);
    }

    pub fn dim(&self) -> usize {
        self.zero.dim()
    }

    fn apply_2x2(&mut self, axis: usize, filter: BitSet, matrix: Matrix2<Complex<f64>>) {
        self.zero.filter_propagate(axis, filter, 0, &|low, high| {
            let uv: Vector2<_> = matrix * Vector2::new(*low, *high);
            *low = uv[0];
            *high = uv[1];
        });
    }
}

impl QuantumState for CubeVector {
    type BasisValue = Complex<f64>;

    fn collapse(&self) -> usize {
        let ri = Arc::new(RwLock::new(rand::random_range(0.0..1.0)));
        let collapsed = Arc::new(RwLock::new(None));
        self.for_each(|state, val| {
            let prob = val.norm_sqr();
            let mut ri_guard = write!(ri);
            *ri_guard -= prob;
            if *ri_guard <= 0.0 {
                write!(collapsed).get_or_insert(state);
            }
        });

        read!(collapsed).unwrap_or(0)
    }

    fn basis_value(&self, basis: usize) -> Self::BasisValue {
        self.zero.at(basis.into(), 0)
    }
}

type Locker<T> = Arc<RwLock<T>>;
type V = Complex<f64>;

#[derive(Debug)]
pub struct SendPtr<T>(Option<NonNull<T>>);

impl<T> SendPtr<T> {
    pub unsafe fn construct(value: T) -> Self {
        let boxed = Box::new(value);
        let ptr = NonNull::from(Box::leak(boxed));
        Self(Some(ptr))
    }

    pub unsafe fn destroy(ptr: Self) {
        unsafe {
            drop(Box::from_raw(
                ptr.0.expect("Expected non null pointer").as_ptr(),
            ));
        }
    }
}

impl<T> Deref for SendPtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.expect("Expected non null pointer").as_ref() }
    }
}
impl<T> DerefMut for SendPtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.expect("Expected non null pointer").as_mut() }
    }
}

impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        Self(Some(NonNull::clone(
            &self.0.expect("Expected non null pointer"),
        )))
    }
}
impl<T> Copy for SendPtr<T> {}

unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}

#[derive(Debug, Clone)]
pub struct Vertex {
    amplitude: V,
    next_vertex: Vec<SendPtr<Self>>,
}

impl Vertex {
    fn new(n: usize) -> Self {
        if n == 0 {
            Self {
                amplitude: cart!(0),
                next_vertex: Vec::with_capacity(0),
            }
        } else {
            let mut ret = Vertex::new(n - 1);
            unsafe {
                ret.ascend_with(SendPtr::construct(Vertex::new(n - 1)));
            };
            ret
        }
    }

    unsafe fn destroy(&mut self, path: BitSet, path_offset: usize, start_axis: usize) {
        for i in start_axis..self.dim() {
            let mut next = path;
            let path_offset = path_offset + i - start_axis;
            next.set(path_offset);
            let mut vertex = Self::vertex_mut(&self.next_vertex, i);
            unsafe { Vertex::destroy(&mut *vertex, next, path_offset + 1, i) };
            unsafe {
                SendPtr::destroy(vertex);
            }
        }
        self.next_vertex.clear();
    }

    fn for_each<F: Fn(usize, V) + std::marker::Send + std::marker::Sync>(
        &self,
        path: BitSet,
        path_offset: usize,
        start_axis: usize,
        f: &F,
    ) {
        f(*path, self.amplitude);
        (start_axis..self.dim()).into_par_iter().for_each(move |i| {
            let mut next = path;
            let path_offset = path_offset + i - start_axis;
            next.set(path_offset);
            Self::vertex_mut(&self.next_vertex, i).for_each(next, path_offset + 1, i, f);
        });
    }

    fn map<F: Fn(usize, &mut V) + std::marker::Send + std::marker::Sync>(
        &mut self,
        path: BitSet,
        path_offset: usize,
        start_axis: usize,
        f: &F,
    ) {
        f(*path, &mut self.amplitude);
        (start_axis..self.dim()).into_par_iter().for_each(move |i| {
            let mut next = path;
            let path_offset = path_offset + i - start_axis;
            next.set(path_offset);
            Self::vertex_mut(&self.next_vertex, i).map(next, path_offset + 1, i, f);
        });
    }

    fn propagate<F: Fn(&mut V, &mut V) + std::marker::Send + std::marker::Sync>(
        &mut self,
        axis: usize,
        start_axis: usize,
        f: &F,
    ) {
        f(
            &mut self.amplitude,
            &mut Self::vertex_mut(&self.next_vertex, axis).amplitude,
        );
        let slf: &Self = self;
        rayon::join(
            || {
                (start_axis..axis).into_par_iter().for_each(move |i| {
                    Self::vertex_mut(&slf.next_vertex, i).propagate(axis - 1, i, f);
                });
            },
            || {
                ((axis + 1).max(start_axis)..slf.dim())
                    .into_par_iter()
                    .for_each(move |i| {
                        Self::vertex_mut(&slf.next_vertex, i).propagate(axis, i, f);
                    });
            },
        );
    }

    #[cfg(not(doctest))]
    /// ```
    /// assert_eq!(*filter & (1 << axis), 0);
    /// ```
    fn filter_propagate<F: Fn(&mut V, &mut V) + std::marker::Send + std::marker::Sync>(
        &mut self,
        axis: usize,
        filter: BitSet,
        filter_offset: usize,
        f: &F,
    ) {
        let mut filter = filter;
        let mut axis = axis;
        debug_assert_eq!(*filter & (1 << axis), 0);

        if filter.is_empty() {
            self.propagate(axis, 0, &f);
        }

        for i in filter_offset..self.dim() {
            if filter[i] {
                filter.erase(i);
                if i < axis {
                    axis -= 1;
                }
                Self::vertex_mut(&self.next_vertex, i).filter_propagate(axis, filter, i, f);
                return;
            }
        }
    }

    #[cfg(not(doctest))]
    /// ```
    /// assert!(axis_1 < axis_2)
    /// ```
    fn swap(&mut self, axis_1: usize, axis_2: usize, start_axis: usize) {
        debug_assert!(axis_1 < axis_2);
        mem::swap(
            &mut Self::vertex_mut(&self.next_vertex, axis_2).amplitude,
            &mut Self::vertex_mut(&self.next_vertex, axis_1).amplitude,
        );
        rayon::join(
            || {
                (start_axis..axis_1).into_par_iter().for_each(|i| {
                    Self::vertex_mut(&self.next_vertex, i).swap(axis_1 - 1, axis_2 - 1, i);
                });
            },
            || {
                rayon::join(
                    || {
                        ((axis_1 + 1).max(start_axis)..axis_2)
                            .into_par_iter()
                            .for_each(|i| {
                                Self::vertex_mut(&self.next_vertex, i).swap(axis_1, axis_2 - 1, i);
                            });
                    },
                    || {
                        ((axis_2 + 1).max(start_axis)..self.dim())
                            .into_par_iter()
                            .for_each(|i| {
                                Self::vertex_mut(&self.next_vertex, i).swap(axis_1, axis_2, i);
                            });
                    },
                );
            },
        );
    }

    #[cfg(not(doctest))]
    /// ```
    /// assert_eq!(*filter & (1 << axis_1), 0)
    /// assert_eq!(*filter & (1 << axis_2), 0)
    /// assert!(axis_1 < axis_2)
    /// ```
    fn filter_swap(&mut self, axis_1: usize, axis_2: usize, filter: BitSet, filter_offset: usize) {
        debug_assert_eq!(*filter & (1 << axis_1), 0);
        debug_assert_eq!(*filter & (1 << axis_2), 0);
        debug_assert!(axis_1 < axis_2);
        let mut filter = filter;
        let mut axis_1 = axis_1;
        let mut axis_2 = axis_2;

        if filter.is_empty() {
            self.swap(axis_1, axis_2, 0);
        }

        for i in filter_offset..self.dim() {
            if filter[i] {
                filter.erase(i);
                if i < axis_1 {
                    axis_1 -= 1;
                    axis_2 -= 1;
                } else if i < axis_2 {
                    axis_2 -= 1;
                }
                Self::vertex_mut(&mut self.next_vertex, i).filter_swap(axis_1, axis_2, filter, i);
                return;
            }
        }
    }

    fn dim(&self) -> usize {
        self.next_vertex.len()
    }

    fn vertex(next_vertex: &Vec<SendPtr<Self>>, axis: usize) -> SendPtr<Self> {
        next_vertex[next_vertex.len() - axis - 1]
        //read!(next_vertex[next_vertex.len() - axis - 1])
    }

    fn vertex_mut(next_vertex: &Vec<SendPtr<Self>>, axis: usize) -> SendPtr<Self> {
        next_vertex[next_vertex.len() - axis - 1]
        // write!(next_vertex[next_vertex.len() - axis - 1])
    }

    /// Does nothing if self and cube are not the same dimension
    unsafe fn ascend_with(&mut self, cube: SendPtr<Self>) {
        if self.dim() != cube.dim() {
            return;
        }
        for i in 0..self.dim() {
            unsafe { self.next_vertex[i].ascend_with(cube.next_vertex[i]) };
        }
        self.next_vertex.push(cube);
    }

    fn at(&self, path: BitSet, path_offset: usize) -> Complex<f64> {
        let mut path = path;
        let mut path_offset = path_offset;
        while !path.is_empty() {
            if path[path_offset] {
                path.erase(path_offset);
                return Self::vertex(&self.next_vertex, path_offset).at(path, path_offset);
            }
            path_offset += 1;
        }
        self.amplitude
    }
}

impl Drop for Vertex {
    fn drop(&mut self) {
        unsafe { self.destroy(0.into(), 0, 0) };
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::{Arc, RwLock},
    };

    use crate::{
        cart, common_test,
        cube_map2::{CubeSimulator, Vertex},
        ext::BitSet,
    };

    #[test]
    /// Assert that foreach only visits each index once
    fn foreach() {
        for n in 0..5 {
            let t = Vertex::new(n);
            let st = Arc::new(RwLock::new(HashSet::new()));
            t.for_each(0.into(), 0, 0, &|i, _| {
                write!(st).insert(i);
            });
            assert_eq!(read!(st).len(), (1 << n));
        }
    }

    #[test]
    /// Assert that propagate only visits each index once
    fn propagate_id() {
        let tt = |ax, n| {
            let mut t = Vertex::new(n);
            let st = Arc::new(RwLock::new(0));
            t.propagate(ax, 0, &mut |src, dst| {
                *write!(st) += 1;
                *src += cart!(1);
                *dst += cart!(1);
            });
            assert_eq!(*read!(st), (1 << (n - 1)));
            t.for_each(0.into(), 0, 0, &mut |_, v| assert!(v == cart!(1)));
        };

        for n in 1..=5 {
            for ax in 0..n {
                tt(ax, n);
            }
        }
    }

    #[test]
    /// Assert that propagate only visits each filtered index once
    fn propagate_filter() {
        let tt = |ax, n, filter: BitSet, n_ctrls| {
            let mut t = Vertex::new(n);
            let st = Arc::new(RwLock::new(0));
            t.filter_propagate(ax, filter, 0, &|src, dst| {
                *write!(st) += 1;
                *src += cart!(1);
                *dst += cart!(1);
            });
            assert_eq!(*read!(st), (1 << (n - 1)) >> n_ctrls);
            t.for_each(0.into(), 0, 0, &mut |i, v| {
                if i & *filter == *filter {
                    assert_eq!(v, cart!(1))
                } else {
                    assert_eq!(v, cart!(0))
                }
            });
        };

        for n in 1..=5 {
            for filter in 0..(1 << n) {
                for ax in 0..n {
                    if filter | (1 << ax) == filter {
                        continue;
                    }
                    tt(ax, n, filter.into(), filter.count_ones());
                }
            }
        }
    }

    #[test]
    /// Assert that propagate only visits each filtered index once
    fn swap() {
        let tt = |ax1, ax2, n| {
            let mut t = Vertex::new(n);
            t.map(0.into(), 0, 0, &mut |i, v| *v = cart!(i));
            t.swap(ax1, ax2, 0);
            t.for_each(0.into(), 0, 0, &mut |i, v| {
                let mut ch = BitSet::from(i);
                ch.swap(ax1, ax2);
                assert_eq!(v, cart!(*ch))
            });
        };

        for n in 1..=5 {
            for axis_1 in 0..n {
                for axis_2 in (axis_1 + 1)..n {
                    tt(axis_1, axis_2, n)
                }
            }
        }
    }

    #[test]
    /// Assert that propagate only visits each filtered index once
    fn swap_filter() {
        let tt = |ax1, ax2, n, filter: BitSet| {
            let mut t = Vertex::new(n);
            t.map(0.into(), 0, 0, &mut |i, v| *v = cart!(i));
            t.filter_swap(ax1, ax2, filter, 0);
            t.for_each(0.into(), 0, 0, &mut |i, v| {
                if *filter & i == *filter {
                    let mut ch = BitSet::from(i);
                    ch.swap(ax1, ax2);
                    assert_eq!(v, cart!(*ch))
                } else {
                    assert_eq!(v, cart!(i));
                }
            });
        };

        for n in 1..=5 {
            for filter in 0..(1 << n) {
                for axis_1 in 0..n {
                    if filter | (1 << axis_1) == filter {
                        continue;
                    }
                    for axis_2 in (axis_1 + 1)..n {
                        if filter | (1 << axis_2) == filter {
                            continue;
                        }
                        tt(axis_1, axis_2, n, filter.into())
                    }
                }
            }
        }
    }

    #[test]
    pub fn apply_gates() {
        common_test::apply_gates::<CubeSimulator>();
    }

    #[test]
    pub fn double_sub() {
        common_test::double_sub::<CubeSimulator>();
    }

    #[test]
    pub fn deep_sub() {
        common_test::deep_sub::<CubeSimulator>();
    }

    #[test]
    pub fn hybrid_test() {
        common_test::hybrid_test::<CubeSimulator>();
    }

    #[test]
    pub fn register_test() {
        common_test::register_test::<CubeSimulator>();
    }

    #[test]
    pub fn test_measure_overwrites_with_zero() {
        common_test::test_measure_overwrites_with_zero::<CubeSimulator>();
    }

    #[test]
    pub fn test_reset() {
        common_test::test_reset::<CubeSimulator>();
    }

    #[test]
    pub fn test_reset_with_shared_scratch_register() {
        common_test::test_reset_with_shared_scratch_register::<CubeSimulator>();
    }

    #[test]
    pub fn deep_ctrl_sub() {
        common_test::deep_ctrl_sub::<CubeSimulator>();
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CubeError {}
