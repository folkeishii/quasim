pub struct IndexedState {
    pub index: usize,
    pub state: String,
}

#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("{0} is outside the state size of {1}")]
    IllegalState(usize, usize),
}
