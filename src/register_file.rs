use std::{
    collections::{HashMap, HashSet},
    ops::{Index, IndexMut},
};

#[derive(Debug, Clone, Default)]
/// Map containing named registers
pub struct RegisterFile<T>(HashMap<String, T>);

impl<T: Default> From<&HashSet<String>> for RegisterFile<T> {
    fn from(value: &HashSet<String>) -> Self {
        let map = value
            .into_iter()
            .map(|name| (name.clone(), T::default()))
            .collect();

        Self(map)
    }
}

impl<T> Index<&str> for RegisterFile<T> {
    type Output = T;

    fn index(&self, index: &str) -> &Self::Output {
        self.0
            .get(index)
            .expect(&format!("Unknown register {}", index))
    }
}
impl<T> IndexMut<&str> for RegisterFile<T> {
    fn index_mut(&mut self, index: &str) -> &mut Self::Output {
        self.0
            .get_mut(index)
            .expect(&format!("Unknown register {}", index))
    }
}

impl<T: Clone> From<&RegisterFile<T>> for HashMap<String, T> {
    fn from(value: &RegisterFile<T>) -> Self {
        value.0.clone()
    }
}
