use rayon::prelude::*;
use std::{
    mem,
    ops::Index,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
    thread,
};

use nalgebra::{Complex, Matrix2, Vector2};

use crate::{
    cart,
    circuit::{Circuit, CircuitBehaviour, HybridCircuit, pc::CircuitPc},
    expr_dsl::{BitExpr, BoolExpr},
    ext::{BitSet, TargetIter},
    gate::{Gate, GateType, QBits},
    register_file::RegisterFile,
    simulator::{DebuggableSimulator, StoredCircuitSimulator},
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
    pub fn collapse_peek(&self) -> usize {
        let ri = Arc::new(RwLock::new(rand::random_range(0.0..1.0)));
        let collapsed = Arc::new(RwLock::new(None));
        self.state_vector.for_each(|state, val| {
            let prob = val.norm_sqr();
            let mut ri_guard = write!(ri);
            *ri_guard -= prob;
            if *ri_guard <= 0.0 {
                write!(collapsed).get_or_insert(state);
            }
        });

        read!(collapsed).unwrap_or(0)
    }

    fn handle_gate(&mut self, gate: Gate) {
        self.pc.increment();
        self.state_vector.apply_gate(gate);
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
        let collapsed = self.collapse_peek();
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

impl DebuggableSimulator for CubeSimulator {
    type Storage = CubeVector;
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

    fn current_state(&self) -> &Self::Storage {
        &self.state_vector
    }

    fn collapse_peek(&self) -> usize {
        self.collapse_peek()
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

impl StoredCircuitSimulator for CubeSimulator {
    type B = HybridCircuit;

    fn circuit(&self) -> &Circuit<Self::B> {
        &self.circuit
    }

    fn circuit_mut(&mut self) -> &mut Circuit<Self::B> {
        &mut self.circuit
    }
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

type Locker<T> = Arc<RwLock<T>>;
type V = Complex<f64>;

#[derive(Debug, Clone)]
pub struct Vertex {
    amplitude: V,
    next_vertex: Vec<Locker<Self>>,
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
            ret.ascend_with(Arc::new(RwLock::new(Vertex::new(n - 1))));
            ret
        }
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

    /// ```
    /// assert!(axis_1 < axis_2)
    /// ```
    fn swap(&mut self, axis_1: usize, axis_2: usize, start_axis: usize) {
        debug_assert!(axis_1 < axis_2);
        mem::swap(
            &mut Self::vertex_mut(&self.next_vertex, axis_1).amplitude,
            &mut Self::vertex_mut(&self.next_vertex, axis_2).amplitude,
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
                Self::vertex_mut(&self.next_vertex, i).filter_swap(axis_1, axis_2, filter, i);
                return;
            }
        }
    }

    fn dim(&self) -> usize {
        self.next_vertex.len()
    }

    fn vertex(next_vertex: &Vec<Locker<Self>>, axis: usize) -> RwLockReadGuard<'_, Self> {
        read!(next_vertex[next_vertex.len() - axis - 1])
    }

    fn vertex_mut(next_vertex: &Vec<Locker<Self>>, axis: usize) -> RwLockWriteGuard<'_, Self> {
        write!(next_vertex[next_vertex.len() - axis - 1])
    }

    /// Does nothing if self and cube are not the same dimension
    fn ascend_with(&mut self, cube: Locker<Self>) {
        let cube_guard = read!(cube);
        if self.dim() != cube_guard.dim() {
            return;
        }
        for i in 0..self.dim() {
            write!(self.next_vertex[i]).ascend_with(Arc::clone(&cube_guard.next_vertex[i]));
        }
        drop(cube_guard);
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

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::{Arc, RwLock},
    };

    use crate::{
        cart,
        cube_map2::{V, Vertex},
        ext::BitSet,
    };

    #[test]
    /// Assert that foreach only visits each index once
    fn foreach() {
        let t = Vertex::new(3);
        let st = Arc::new(RwLock::new(HashSet::new()));
        t.for_each(0.into(), 0, 0, &|i, _| {
            write!(st).insert(i);
        });
        assert_eq!(read!(st).len(), (1 << 3));
        write!(st).clear();
        let t = Vertex::new(4);
        t.for_each(0.into(), 0, 0, &mut |i, _| {
            write!(st).insert(i);
        });
        assert_eq!(read!(st).len(), (1 << 4));
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
}

#[derive(Debug, thiserror::Error)]
pub enum CubeError {}
