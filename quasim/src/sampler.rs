use crate::{
    circuit::{Circuit, CircuitBehaviour, HybridCircuit},
    simulator::{QuantumState, Simulator, StoredRegisters},
};

pub trait Sampler<T: Simulator> {
    type CircuitBehaviour: CircuitBehaviour;
    type Output;

    fn circuit(&self) -> &Circuit<Self::CircuitBehaviour>;
    fn sample(&self, simulator: &T) -> Self::Output;
}

/// # CircuitSampler
/// Samples all qubits in the circuit as an unsigned integer
pub struct CircuitSampler<B: CircuitBehaviour> {
    circuit: Circuit<B>,
}
impl<T: Simulator, B: CircuitBehaviour> Sampler<T> for CircuitSampler<B> {
    type CircuitBehaviour = B;
    type Output = usize;

    fn circuit(&self) -> &Circuit<Self::CircuitBehaviour> {
        &self.circuit
    }

    fn sample(&self, simulator: &T) -> Self::Output {
        simulator.state().collapse()
    }
}

/// # QubitSampler
/// Specifies a single qubit to be sampled
pub struct QubitSampler<B: CircuitBehaviour> {
    circuit: Circuit<B>,
    bit: usize,
}
impl<T: Simulator, B: CircuitBehaviour> Sampler<T> for QubitSampler<B> {
    type CircuitBehaviour = B;
    type Output = usize;

    fn circuit(&self) -> &Circuit<Self::CircuitBehaviour> {
        &self.circuit
    }

    fn sample(&self, simulator: &T) -> Self::Output {
        (simulator.state().collapse() >> self.bit) & 1
    }
}

/// # QubitsSampler
/// Specifies a vector of qubits to be sampled
pub struct QubitsSampler<B: CircuitBehaviour> {
    circuit: Circuit<B>,
    bits: Vec<usize>,
}
impl<T: Simulator, B: CircuitBehaviour> Sampler<T> for QubitsSampler<B> {
    type CircuitBehaviour = B;
    type Output = Vec<usize>;

    fn circuit(&self) -> &Circuit<Self::CircuitBehaviour> {
        &self.circuit
    }

    fn sample(&self, simulator: &T) -> Self::Output {
        let collapsed_result = simulator.state().collapse();

        let mut result = Vec::new();

        for &bit in &self.bits {
            result.push((collapsed_result >> bit) & 1);
        }

        result
    }
}

/// # RegSampler
/// Specifies a specific register to be sampled
pub struct RegisterSampler {
    circuit: Circuit<HybridCircuit>,
    reg: String,
}
impl<T: Simulator + StoredRegisters> Sampler<T> for RegisterSampler {
    type CircuitBehaviour = HybridCircuit;
    type Output = usize;

    fn circuit(&self) -> &Circuit<Self::CircuitBehaviour> {
        &self.circuit
    }

    fn sample(&self, simulator: &T) -> Self::Output {
        simulator.register(&self.reg).read()
    }
}

impl<B> Circuit<B>
where
    B: CircuitBehaviour,
    Circuit<B>: Clone,
{
    pub fn sample(&self) -> CircuitSampler<B> {
        CircuitSampler {
            circuit: self.clone(),
        }
    }

    pub fn sample_bit(&self, bit: usize) -> QubitSampler<B> {
        QubitSampler {
            circuit: self.clone(),
            bit,
        }
    }

    pub fn sample_bits(&self, bits: &[usize]) -> QubitsSampler<B> {
        QubitsSampler {
            circuit: self.clone(),
            bits: bits.to_vec(),
        }
    }
}

impl Circuit<HybridCircuit> {
    pub fn sample_reg(&self, reg: &str) -> RegisterSampler {
        RegisterSampler {
            circuit: self.clone(),
            reg: reg.to_owned(),
        }
    }
}
