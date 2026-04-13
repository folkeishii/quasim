use std::{
    mem,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use nalgebra::Complex;

use crate::{cart, ext::BitSet};


type Locker<T> = Arc<RwLock<T>>;
type V = Complex<f64>;
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

    fn for_each<F: FnMut(usize, V)>(
        &self,
        path: BitSet,
        path_offset: usize,
        start_axis: usize,
        f: &mut F,
    ) {
        let mut path_offset = path_offset;
        f(*path, self.amplitude);
        for i in start_axis..self.dim() {
            let mut next = path;
            next.set(path_offset);
            Self::vertex_mut(&self.next_vertex, i).for_each(next, path_offset + 1, i, f);
            path_offset += 1;
        }
    }

    fn map<F: FnMut(usize, &mut V)>(
        &mut self,
        path: BitSet,
        path_offset: usize,
        start_axis: usize,
        f: &mut F,
    ) {
        let mut path_offset = path_offset;
        f(*path, &mut self.amplitude);
        for i in start_axis..self.dim() {
            let mut next = path;
            next.set(path_offset);
            Self::vertex_mut(&self.next_vertex, i).map(next, path_offset + 1, i, f);
            path_offset += 1;
        }
    }

    fn propagate<F: FnMut(&mut V, &mut V)>(&mut self, axis: usize, start_axis: usize, f: &mut F) {
        f(
            &mut self.amplitude,
            &mut Self::vertex_mut(&self.next_vertex, axis).amplitude,
        );
        for i in start_axis..axis {
            Self::vertex_mut(&self.next_vertex, i).propagate(axis - 1, i, f);
        }
        for i in (axis + 1).max(start_axis)..self.dim() {
            Self::vertex_mut(&self.next_vertex, i).propagate(axis, i, f);
        }
    }

    /// ```
    /// assert_eq!(*filter & (1 << axis), 0);
    /// ```
    fn filter_propagate<F: FnMut(&mut V, &mut V)>(
        &mut self,
        axis: usize,
        filter: BitSet,
        filter_offset: usize,
        mut f: F,
    ) {
        let mut filter = filter;
        let mut axis = axis;
        debug_assert_eq!(*filter & (1 << axis), 0);

        if filter.is_empty() {
            self.propagate(axis, 0, &mut f);
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
        for i in start_axis..axis_1 {
            Self::vertex_mut(&self.next_vertex, i).swap(axis_1 - 1, axis_2 - 1, i);
        }
        for i in (axis_1 + 1).max(start_axis)..axis_2 {
            Self::vertex_mut(&self.next_vertex, i).swap(axis_1, axis_2 - 1, i);
        }
        for i in (axis_2 + 1).max(start_axis)..self.dim() {
            Self::vertex_mut(&self.next_vertex, i).swap(axis_1, axis_2, i);
        }
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
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::{
        cart,
        cube_map2::{V, Vertex},
        ext::BitSet,
    };

    #[test]
    /// Assert that foreach only visits each index once
    fn foreach() {
        let t = Vertex::new(3);
        let st = &mut HashSet::new();
        t.for_each(0.into(), 0, 0, &mut |i, _| {
            st.insert(i);
        });
        assert_eq!(st.len(), (1 << 3));
        st.clear();
        let t = Vertex::new(4);
        t.for_each(0.into(), 0, 0, &mut |i, _| {
            st.insert(i);
        });
        assert_eq!(st.len(), (1 << 4));
    }

    #[test]
    /// Assert that propagate only visits each index once
    fn propagate_id() {
        let tt = |ax, n| {
            let mut t = Vertex::new(n);
            let st = &mut 0;
            t.propagate(ax, 0, &mut |src, dst| {
                *st += 1;
                *src += cart!(1);
                *dst += cart!(1);
            });
            assert_eq!(*st, (1 << (n - 1)));
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
            let st = &mut 0;
            t.filter_propagate(ax, filter, 0, |src, dst| {
                *st += 1;
                *src += cart!(1);
                *dst += cart!(1);
            });
            assert_eq!(*st, (1 << (n - 1)) >> n_ctrls);
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
