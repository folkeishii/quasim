use crate::simulator::{QuantumState, Simulator, StoredRegisters};

pub trait Sampler<S: Simulator> {
    type Output;

    fn sample(&self, simulator: &S) -> Self::Output;
}

impl<'a, T: Sampler<S>, S: Simulator> Sampler<S> for &'a T {
    type Output = T::Output;

    fn sample(&self, simulator: &S) -> Self::Output {
        T::sample(self, simulator)
    }
}

// -------------------------------- //
// Samplers                         //
// -------------------------------- //

#[derive(Debug, Clone, Copy)]
/// # CircuitSampler
/// Samples all qubits in the circuit as an unsigned integer
pub struct CircuitSampler;
impl<S: Simulator> Sampler<S> for CircuitSampler {
    type Output = usize;

    fn sample(&self, simulator: &S) -> Self::Output {
        simulator.state().collapse()
    }
}

#[derive(Debug, Clone, Copy)]
/// # QubitSampler
/// Specifies a single qubit to be sampled
pub struct QubitSampler {
    bit: usize,
}
impl QubitSampler {
    pub fn new(bit: usize) -> Self {
        Self { bit: bit }
    }
}
impl<S: Simulator> Sampler<S> for QubitSampler {
    type Output = usize;

    fn sample(&self, simulator: &S) -> Self::Output {
        (simulator.state().collapse() >> self.bit) & 1
    }
}

#[derive(Debug, Clone)]
/// # QubitsSampler
/// Specifies a vector of qubits to be sampled
pub struct QubitsSampler {
    bits: Vec<usize>,
}
impl QubitsSampler {
    pub fn new<I: IntoIterator<Item = usize>>(bits: I) -> Self {
        Self {
            bits: bits.into_iter().collect(),
        }
    }
}
impl<S: Simulator> Sampler<S> for QubitsSampler {
    type Output = Vec<usize>;

    fn sample(&self, simulator: &S) -> Self::Output {
        let collapsed_result = simulator.state().collapse();

        let mut result = Vec::new();

        for &bit in &self.bits {
            result.push((collapsed_result >> bit) & 1);
        }

        result
    }
}

#[derive(Debug, Clone)]
/// # RegisterSampler
/// Specifies a specific register to be sampled
pub struct RegisterSampler {
    reg: String,
}
impl RegisterSampler {
    pub fn new<S: Into<String>>(reg: S) -> Self {
        Self { reg: reg.into() }
    }
}
impl<S: Simulator + StoredRegisters> Sampler<S> for RegisterSampler {
    type Output = usize;

    fn sample(&self, simulator: &S) -> Self::Output {
        simulator.register(&self.reg).read()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        circuit::Circuit,
        sampler::{CircuitSampler, QubitSampler, QubitsSampler, RegisterSampler},
        simulator::Sampleable,
        sv_simulator::StateVectorSimulator,
    };

    #[test]
    fn sample_circuit() {
        let circuit = Circuit::new(5).x(1).x(3);
        assert!(
            StateVectorSimulator::sample(circuit, CircuitSampler, 10)
                .unwrap()
                .fold(true, |acc, it| { acc && (it ^ 0b01010 == 0) })
        )
    }

    #[test]
    fn sample_qubit() {
        let circuit = Circuit::new(5).x(1).x(3);
        assert!(
            StateVectorSimulator::sample(circuit, QubitSampler::new(1), 10)
                .unwrap()
                .fold(true, |acc, it| { acc && it == 1 })
        )
    }

    #[test]
    fn sample_qubits() {
        let circuit = Circuit::new(5).x(1).x(3);
        assert!(
            StateVectorSimulator::sample(circuit, QubitsSampler::new([1, 3]), 10)
                .unwrap()
                .fold(true, |acc, it| { acc && it == [1, 1] })
        )
    }

    #[test]
    fn sample_reg() {
        let circuit = Circuit::new(5).new_reg("res", 5).x(1).measure("res").x(3);
        assert!(
            StateVectorSimulator::sample(circuit, RegisterSampler::new("res"), 10)
                .unwrap()
                .fold(true, |acc, it| { acc && it == 0b00010 })
        )
    }
}
