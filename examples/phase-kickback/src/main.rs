use nalgebra::{Complex, DVector};
use quasim::circuit::Circuit;
use quasim::simulator::{BuildSimulator, DebuggableSimulator};
use quasim::sv_simulator::SVSimulatorDebugger;
use std::f64::consts::PI;

#[derive(Debug)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() <= 1e-6
            && (self.y - other.y).abs() <= 1e-6
            && (self.z - other.z).abs() <= 1e-6
    }
}

/// Get bloch vector for second qubit
///
/// References:
/// https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/multiple-systems
/// https://www.sciencedirect.com/science/article/pii/S0375960101004558
fn get_bloch_vector(state_vector: &DVector<Complex<f64>>) -> Point {
    let p = state_vector * state_vector.adjoint();

    let a = p[(0, 0)] + p[(1, 1)];
    let b = p[(0, 2)] + p[(1, 3)];
    let c = p[(2, 0)] + p[(3, 1)];

    Point {
        x: 2.0 * b.re,
        y: 2.0 * c.im,
        z: 2.0 * a.re - 1.0,
    }
}

#[allow(dead_code)]
fn before_phase_kickback() -> Point {
    let circuit = Circuit::new(2).h(0).x(1);

    let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
    sim.continue_until(None);

    get_bloch_vector(sim.current_state())
}

#[allow(dead_code)]
fn after_phase_kickback() -> Point {
    let circuit = Circuit::new(2)
        .h(0)
        .x(1)
        .cu(0.0, 0.0, PI / 2.0, &[0], 1)
        .h(0);

    let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
    sim.continue_until(None);

    get_bloch_vector(sim.current_state())
}

fn main() {
    let circuit = Circuit::new(2)
        .h(0)
        .x(1)
        .cu(0.0, 0.0, PI / 2.0, &[0], 1)
        .h(0);

    let mut sim = SVSimulatorDebugger::build(circuit).unwrap();
    sim.continue_until(None);

    println!("{}", sim.current_state());
    println!("{:?}", get_bloch_vector(sim.current_state()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_kickback() {
        assert_eq!(before_phase_kickback(), after_phase_kickback());
    }
}
