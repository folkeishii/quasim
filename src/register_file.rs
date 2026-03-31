use std::fmt::Display;
use std::{
    collections::HashMap,
    ops::{Index, IndexMut},
};

#[derive(Debug, Clone, thiserror::Error)]
pub enum RegisterError {
    #[error("tried to create register with invalid size {0}")]
    InitSize(usize),
    #[error("tried to write to out of bounds bit {0}")]
    WriteBitError(usize),
    #[error("tried to write invalid value {0}, expected 0 or 1")]
    WriteBitValueError(usize),
}

#[derive(Debug, Clone, Copy)]
pub struct Register {
    value: usize,
    size: usize,
}

impl Register {
    /// Expects size in range 0 to 64
    pub fn new(size: usize) -> Result<Self, RegisterError> {
        if size > 64 {
            Err(RegisterError::InitSize(size))
        } else {
            Ok(Self { value: 0, size })
        }
    }

    /// Expects value as 0 or 1
    pub fn write_bit(&mut self, bit: usize, value: usize) -> Result<usize, RegisterError> {
        if bit >= self.size {
            return Err(RegisterError::WriteBitError(bit));
        }
        if !(value == 0 || value == 1) {
            return Err(RegisterError::WriteBitValueError(value));
        }

        // Reset bit pos
        self.value &= !(1 << bit);
        // Write new bit value
        self.value |= value << bit;

        Ok(self.value)
    }

    pub fn write(&mut self, value: usize) {
        // Write mask is 1's in writable bit positions
        let write_mask = (1 << self.size) - 1;
        self.value = value & write_mask;
    }

    pub fn read_bit(&self, bit: usize) -> usize {
        (self.value >> bit) & 1
    }

    pub fn read(&self) -> usize {
        self.value
    }
}

impl Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:0>width$b}", self.value, width = self.size)
    }
}

#[derive(Debug, Clone, Default)]
/// Map containing named registers
pub struct RegisterFile(HashMap<String, Register>);

impl RegisterFile {
    pub fn get(&self, key: &str) -> Option<&Register> {
        self.0.get(key)
    }
}

impl TryFrom<&HashMap<String, usize>> for RegisterFile {
    type Error = RegisterError;

    fn try_from(value: &HashMap<String, usize>) -> Result<Self, Self::Error> {
        let mut reg_map = HashMap::new();

        for (name, &size) in value {
            let new_reg = Register::new(size)?;

            reg_map.insert(name.clone(), new_reg);
        }

        Ok(Self(reg_map))
    }
}

impl Index<&str> for RegisterFile {
    type Output = Register;

    fn index(&self, index: &str) -> &Self::Output {
        self.0
            .get(index)
            .expect(&format!("Unknown register {}", index))
    }
}
impl IndexMut<&str> for RegisterFile {
    fn index_mut(&mut self, index: &str) -> &mut Self::Output {
        self.0
            .get_mut(index)
            .expect(&format!("Unknown register {}", index))
    }
}

impl From<&RegisterFile> for HashMap<String, Register> {
    fn from(value: &RegisterFile) -> Self {
        value.0.clone()
    }
}

impl Display for RegisterFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let key_width = self.0.keys().map(|k| k.len()).max().unwrap_or(1);

        let mut peekable = self.0.iter().peekable();
        while let Some((reg, value)) = peekable.next() {
            write!(f, "{: >key_width$} - {}", reg, value)?;
            if peekable.peek().is_some() {
                writeln!(f)?;
            }
        }

        Ok(())
    }
}
