use std::{
    collections::HashMap,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use crate::gate::QBits;

#[derive(Debug, Clone, Default)]
/// Map containing named registers
pub struct RegisterFile<T>(RegisterFileRef<T>);
impl<T> RegisterFile<T> {
    /// Add register
    pub fn add(&mut self, name: String) -> Option<T>
    where
        T: Default,
    {
        self.0.0.insert(name, T::default())
    }
}
impl<T> Deref for RegisterFile<T> {
    type Target = RegisterFileRef<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for RegisterFile<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone, Default)]
pub struct RegisterFileRef<T>(HashMap<String, T>);
impl<T> RegisterFileRef<T> {
    /// Will panic if name is not registered
    pub fn register(&self, name: &str) -> Option<&T> {
        self.0.get(name)
    }

    /// Will panic if name is not registered
    pub fn register_mut(&mut self, name: &str) -> Option<&mut T> {
        self.0.get_mut(name)
    }
}
impl<T> Index<&str> for RegisterFileRef<T> {
    type Output = T;

    fn index(&self, index: &str) -> &Self::Output {
        self.register(index)
            .expect(&format!("Unknown register {}", index))
    }
}
impl<T> IndexMut<&str> for RegisterFileRef<T> {
    fn index_mut(&mut self, index: &str) -> &mut Self::Output {
        self.register_mut(index)
            .expect(&format!("Unknown register {}", index))
    }
}
