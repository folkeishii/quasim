use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut, Index},
    rc::Rc,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use nalgebra::{Complex, Const, Dim, Dyn};

use crate::{cart, ext::BitSet};

type Locker<T> = Arc<RwLock<T>>;

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
pub struct Vertex<T> {
    amplitude: T,
    next_vertex: Vec<Locker<Vertex<T>>>,
}

impl<T> Vertex<T> {
    fn new(n: usize) -> Self
    where
        T: Default,
    {
        if n == 0 {
            Self {
                amplitude: T::default(),
                next_vertex: Vec::with_capacity(0),
            }
        } else {
            let mut ret = Vertex::new(n - 1);
            ret.ascend_with(Arc::new(RwLock::new(Vertex::new(n - 1))));
            ret
        }
    }

    fn for_each<F: FnMut(usize, &T)>(
        &self,
        path: BitSet,
        path_offset: usize,
        start_axis: usize,
        f: &mut F,
    ) {
        let mut path_offset = path_offset;
        f(*path, &self.amplitude);
        for i in start_axis..self.next_vertex.len() {
            let mut next = path;
            next.set(path_offset);
            Self::vertex_mut(&self.next_vertex, i).for_each(next, path_offset+1, i, f);
            path_offset += 1;
        }
    }

    fn propagate<F: FnMut(&mut T, &mut T)>(&mut self, axis: usize, f: &mut F) {
        let movement_axii = (0..axis)
            .into_iter()
            .chain((axis + 1)..self.next_vertex.len());

        f(
            &mut self.amplitude,
            &mut Self::vertex_mut(&self.next_vertex, axis).amplitude,
        );

        for axis in movement_axii {
            Self::vertex_mut(&self.next_vertex, axis).propagate(axis, f);
        }
    }

    fn propagate_with<F: FnMut(&mut T, &mut T)>(
        &mut self,
        axis: usize,
        filter: BitSet,
        filter_offset: usize,
        f: &mut F,
    ) {
        debug_assert!(*filter | (1 << axis) == *filter);
        let mut axis = axis;
        let mut filter = filter;
        let mut filter_offset = filter_offset;
        while !filter.is_empty() {
            if filter[filter_offset] {
                filter.erase(filter_offset);
                if (filter_offset < axis) {
                    axis -= 1;
                }
                return Self::vertex_mut(&self.next_vertex, filter_offset).propagate_with(
                    axis,
                    filter,
                    filter_offset,
                    f,
                );
            }

            filter_offset += 1;
        }

        self.propagate(axis, f);
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
        for i in 0..self.next_vertex.len() {
            write!(self.next_vertex[i]).ascend_with(Arc::clone(&cube_guard.next_vertex[i]));
        }
        drop(cube_guard);
        self.next_vertex.push(cube);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use nalgebra::Complex;

    use crate::cube_map2::Vertex;

    #[test]
    fn foreach() {
        let t = Vertex::<usize>::new(3);
        let st = &mut HashSet::new();
        t.for_each(0.into(), 0, 0, &mut |i, _| {
            st.insert(i);
        });
        assert_eq!(st.len(), (1<<3));
        st.clear();
        let t = Vertex::<usize>::new(4);
        t.for_each(0.into(), 0, 0, &mut |i, _| {
            st.insert(i);
        });
        assert_eq!(st.len(), (1<<4));
    }
    #[test]
    fn propagate_id() {
        let mut t = Vertex::<usize>::new(3);
        let st = &mut 0;
        t.propagate(0, &mut |_,_| {
            *st += 1;
        });
        assert_eq!(*st, 4);
    }
}
