use crate::{
    circuit::{Circuit, CircuitBehaviour, HybridCircuit},
    simulator::{QuantumState, Sampleable, Simulator, StoredRegisters},
};

pub trait Sampler<S: Simulator> {
    type Output;

    // fn circuit(&self) -> &Circuit<Self::CircuitBehaviour>;
    fn sample(&self, simulator: &S) -> Self::Output;
}

// -------------------------------- //
// Convenience traits               //
//                                  //
// If Sampleable is implemented for //
// Sim then we can add convenience  //
// methods for sim                  //
// -------------------------------- //

/// # CircuitSampleable
pub trait CircuitSampleable<B: CircuitBehaviour>: Sampleable<B, CircuitSampler> {
    fn sample_circuit_once(circuit: Circuit<B>) -> Result<usize, Self::E> {
        Self::sample_once(circuit, CircuitSampler)
    }

    fn sample_circuit(
        circuit: Circuit<B>,
        times: usize,
    ) -> Result<impl Iterator<Item = usize>, Self::E> {
        Self::sample(circuit, CircuitSampler, times)
    }
}
impl<B: CircuitBehaviour, S: Sampleable<B, CircuitSampler>> CircuitSampleable<B> for S {}

/// # QubitSampleable
pub trait QubitSampleable<B: CircuitBehaviour>: Sampleable<B, QubitSampler> {
    fn sample_qubit_once(circuit: Circuit<B>, bit: usize) -> Result<usize, Self::E> {
        Self::sample_once(circuit, QubitSampler::new(bit))
    }

    fn sample_qubit(
        circuit: Circuit<B>,
        bit: usize,
        times: usize,
    ) -> Result<impl Iterator<Item = usize>, Self::E> {
        Self::sample(circuit, QubitSampler::new(bit), times)
    }
}
impl<B: CircuitBehaviour, S: Sampleable<B, QubitSampler>> QubitSampleable<B> for S {}

/// # QubitSampleable
pub trait QubitsSampleable<B, I>: Sampleable<B, QubitsSampler>
where
    B: CircuitBehaviour,
    I: IntoIterator<Item = usize>,
{
    fn sample_qubits_once(circuit: Circuit<B>, bits: I) -> Result<Vec<usize>, Self::E> {
        Self::sample_once(circuit, QubitsSampler::new(bits))
    }

    fn sample_qubits(
        circuit: Circuit<B>,
        bits: I,
        times: usize,
    ) -> Result<impl Iterator<Item = Vec<usize>>, Self::E> {
        Self::sample(circuit, QubitsSampler::new(bits), times)
    }
}
impl<B, I, S> QubitsSampleable<B, I> for S
where
    B: CircuitBehaviour,
    I: IntoIterator<Item = usize>,
    S: Sampleable<B, QubitsSampler>,
{
}

/// # QubitSampleable
pub trait ReqisterSampleable<I>:
    Sampleable<HybridCircuit, RegisterSampler> + StoredRegisters
where
    I: Into<String>,
{
    fn sample_register_once(circuit: Circuit<HybridCircuit>, reg: I) -> Result<usize, Self::E> {
        Self::sample_once(circuit, RegisterSampler::new(reg))
    }

    fn sample_register(
        circuit: Circuit<HybridCircuit>,
        reg: I,
        times: usize,
    ) -> Result<impl Iterator<Item = usize>, Self::E> {
        Self::sample(circuit, RegisterSampler::new(reg), times)
    }
}
impl<I, S> ReqisterSampleable<I> for S
where
    I: Into<String>,
    S: Sampleable<HybridCircuit, RegisterSampler> + StoredRegisters,
{
}

// -------------------------------- //
// Samplers                         //
// -------------------------------- //

/// # CircuitSampler
/// Samples all qubits in the circuit as an unsigned integer
pub struct CircuitSampler;
impl<S: Simulator> Sampler<S> for CircuitSampler {
    type Output = usize;

    fn sample(&self, simulator: &S) -> Self::Output {
        simulator.state().collapse()
    }
}

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

/// # RegSampler
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
        sampler::{CircuitSampleable, QubitSampleable, QubitsSampleable, ReqisterSampleable},
        sv_simulator::SVSimulator,
    };

    #[test]
    fn sample_circuit() {
        let circuit = Circuit::new(5).x(1).x(3);
        assert!(
            SVSimulator::sample_circuit(circuit, 10)
                .unwrap()
                .fold(true, |acc, it| { acc && (it ^ 0b01010 == 0) })
        )
    }

    #[test]
    fn sample_qubit() {
        let circuit = Circuit::new(5).x(1).x(3);
        assert!(
            SVSimulator::sample_qubit(circuit, 1, 10)
                .unwrap()
                .fold(true, |acc, it| { acc && it == 1 })
        )
    }

    #[test]
    fn sample_qubits() {
        let circuit = Circuit::new(5).x(1).x(3);
        assert!(
            SVSimulator::sample_qubits(circuit, [1, 3].into_iter(), 10)
                .unwrap()
                .fold(true, |acc, it| { acc && it == [1, 1] })
        )
    }

    #[test]
    fn sample_reg() {
        let circuit = Circuit::new(5).new_reg("res", 5).x(1).measure("res").x(3);
        assert!(
            SVSimulator::sample_register(circuit, "res", 10)
                .unwrap()
                .fold(true, |acc, it| { acc && it == 0b00010 })
        )
    }
}
