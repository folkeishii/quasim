use std::{
    borrow::Cow,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak},
    usize,
};

use nalgebra::Complex;

use crate::{cart, ext::BitSet};

pub struct NDCube<const N: usize, L> {
    zero: L,
}
impl<const N: usize, L> NDCube<N, L> where L: Locker<Inner = Node<N, L>> {
    pub fn new() -> Self {
        let mut s = Node::construct_incomplete(0);
        s.amplitude = cart!(1);
        Self { zero: L::new(s) }
    }

    fn node(&self, index: usize) -> Option<L> {
        if index == 0 {
            return Some(self.zero.clone())
        }
        self.zero.read().node(index.into(), 0)
    }
}

#[derive(Clone)]
pub struct Node<const N: usize, L: Locker> {
    level: usize,
    edges: [Locked<L>; N],
    amplitude: Complex<f64>,
}
impl<const N: usize, L> Node<N, L>
where
    L: Locker<Inner = Node<N, L>>,
{
    fn node(&self, path: BitSet, offset: usize) -> Option<L> {
        let mut path = path;
        for i in offset..N {
            if path[i] {
                path.erase(i);
                let st = self.edges[i].strong();
                return if path.is_empty() {
                    st.cloned()
                } else if let Some(node) = st {
                    node.read().node(path, i+1)
                } else {
                    None
                }
            }
        }
        None
    }

    fn initialize_first(&mut self, index: usize, root: &NDCube<N, L>) {
        for off in 0..N {}
    }

    fn construct_incomplete(level: usize) -> Self {
        let edges: [_; N] = array_init::array_init(|_| Locked::new_weak());
        Self {
            level,
            edges,
            amplitude: cart!(0),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Locked<L: Locker> {
    Strong(L),
    Weak(L::Weak),
}

impl<L: Locker> Locked<L> {
    pub fn clone_strong(locker: &L) -> Self {
        Self::Strong(locker.clone())
    }
    pub fn new_weak() -> Self {
        Self::Weak(L::Weak::new())
    }
    pub fn clone_weak(locker: &L::Weak) -> Self {
        Self::Weak(locker.clone())
    }
    pub fn strong(&self) -> Option<&L> {
        match self {
            Locked::Strong(st) => Some(st),
            Locked::Weak(_) => None
        }
    }
    pub fn get<'a>(&'a self) -> Cow<'a, L> {
        match self {
            Locked::Strong(st) => Cow::Borrowed(st),
            Locked::Weak(wk) => {
                Cow::Owned(wk.upgrade().expect("Could not obtain reference from weak"))
            }
        }
    }
}

pub trait Locker: Clone {
    type Inner;
    type ReadGuard<'a>: Deref<Target = Self::Inner>
    where
        Self: 'a;
    type WriteGuard<'a>: DerefMut<Target = Self::Inner>
    where
        Self: 'a;
    type Weak: WeakLocker<Inner = Self::Inner, Strong = Self>;

    fn new(value: Self::Inner) -> Self;
    fn read<'a>(&'a self) -> Self::ReadGuard<'a>;
    fn write<'a>(&'a mut self) -> Self::WriteGuard<'a>;
}

pub trait WeakLocker: Clone {
    type Inner;
    type Strong: Locker<Inner = Self::Inner, Weak = Self>;

    fn new() -> Self;
    fn upgrade(&self) -> Option<Self::Strong>;
}

impl<T> Locker for Arc<RwLock<T>> {
    type Inner = T;
    type ReadGuard<'a>
        = RwLockReadGuard<'a, T>
    where
        T: 'a;
    type WriteGuard<'a>
        = RwLockWriteGuard<'a, T>
    where
        T: 'a;
    type Weak = Weak<RwLock<T>>;

    fn new(value: Self::Inner) -> Self {
        Arc::new(RwLock::new(value))
    }

    fn read<'a>(&'a self) -> Self::ReadGuard<'a> {
        RwLock::read(self).expect("Poisoned")
    }

    fn write<'a>(&'a mut self) -> Self::WriteGuard<'a> {
        RwLock::write(self).expect("Poisoned")
    }
}

impl<T> WeakLocker for Weak<RwLock<T>> {
    type Inner = T;
    type Strong = Arc<RwLock<T>>;

    fn new() -> Self {
        Weak::new()
    }

    fn upgrade(&self) -> Option<Self::Strong> {
        Weak::upgrade(self)
    }
}

struct NCrIter {
    next: usize,
    n: usize,
    k: usize,
}
impl NCrIter {
    pub const fn new(n: usize) -> Self {
        Self { next: 1, n, k: 0 }
    }

    fn max(n: usize) -> usize {
        let mut it = Self::new(n);
        let prev = 1;
        while let Some(n) = it.next() {
            if n < prev {
                return prev;
            }
        }
        prev
    }
}
impl Iterator for NCrIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.k > self.n {
            return None;
        }

        let item = self.next;
        self.next *= self.n - self.k;
        self.k += 1;
        self.next /= self.k;
        Some(item)
    }
}

#[cfg(test)]
mod tests {
    use crate::cube_map::NCrIter;

    #[test]
    fn ncr() {
        let c0: &[usize] = &[1];
        let c1: &[usize] = &[1, 1];
        let c2: &[usize] = &[1, 2, 1];
        let c3: &[usize] = &[1, 3, 3, 1];
        let c4: &[usize] = &[1, 4, 6, 4, 1];
        let c5: &[usize] = &[1, 5, 10, 10, 5, 1];
        let correct = [c0, c1, c2, c3, c4, c5];
        for n in 0..=5 {
            let mut it = NCrIter::new(n);
            for k in 0..=n {
                assert_eq!(it.next(), Some(correct[n][k]));
            }
            assert_eq!(it.next(), None)
        }
    }
}
