use crate::{Circuit, Instruction, SimpleSimulator};
use nalgebra::{Complex, DMatrix, DVector, MatrixN, dmatrix};

pub struct DebugSimulator {
    pub states: Vec<DVector<Complex<f32>>>,
}

impl SimpleSimulator for DebugSimulator {
    type E = SimpleError;

    fn build(circuit: Circuit) -> Result<Self, Self::E> {
        let k = circuit.n_qubits;
        let mut init_state = vec![Complex::ZERO; 1 << k];
        init_state[0] = Complex::ONE;

        let mut sim = DebugSimulator {
            states: vec![DVector::from_vec(init_state)],
        };

        for inst in circuit.instructions {
            let mat = DebugSimulator::inst_to_big_matrix(inst, k);
            let v = sim
                .states
                .last()
                .expect("states should have at least the initial state")
                .clone();
            let w = mat * v;
            sim.states.push(w);
        }

        Ok(sim)
    }

    fn final_state(&self) -> DVector<nalgebra::Complex<f32>> {
        return self
            .states
            .last()
            .expect("states should have at least the initial state")
            .clone();
    }

    fn run(&self) -> usize {
        todo!()
    }
}

impl DebugSimulator {
    fn inst_to_big_matrix(inst: Instruction, n_qubits: usize) -> DMatrix<Complex<f32>> {
        match inst {
            Instruction::CNOT(control, target) => DebugSimulator::controlled(
                Instruction::X(0).get_matrix(),
                vec![control],
                target,
                n_qubits,
            ),
            Instruction::X(target) => {
                DebugSimulator::inflate_2x2(inst.get_matrix(), target, n_qubits)
            }
            Instruction::Y(target) => {
                DebugSimulator::inflate_2x2(inst.get_matrix(), target, n_qubits)
            }
            Instruction::Z(target) => {
                DebugSimulator::inflate_2x2(inst.get_matrix(), target, n_qubits)
            }
            Instruction::H(target) => {
                DebugSimulator::inflate_2x2(inst.get_matrix(), target, n_qubits)
            }
        }
    }

    fn inflate_2x2(
        matrix_2x2: DMatrix<Complex<f32>>,
        target: usize,
        n_qubits: usize,
    ) -> DMatrix<Complex<f32>> {
        let mut tensor_factors = vec![DMatrix::<Complex<f32>>::identity(2, 2); n_qubits];
        tensor_factors[target] = matrix_2x2;
        tensor_factors
            .iter()
            .fold(DMatrix::<Complex<f32>>::identity(1, 1), |acc, x| {
                acc.kronecker(x)
            })
    }

    fn controlled(
        matrix_2x2: DMatrix<Complex<f32>>,
        controls: Vec<usize>,
        target: usize,
        n_qubits: usize,
    ) -> DMatrix<Complex<f32>> {
        let ketbra = vec![
            DMatrix::<Complex<f32>>::from_row_slice(
                2,
                2,
                &[Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO],
            ),
            DMatrix::<Complex<f32>>::from_row_slice(
                2,
                2,
                &[Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ONE],
            ),
        ];
        let n_controls = controls.len();
        let n_terms = 1 << n_controls;
        let mut terms = vec![vec![DMatrix::<Complex<f32>>::identity(2, 2); n_qubits]; n_terms];
        for i in 0..n_terms {
            let mut c: usize = 0;
            for control in controls.clone() {
                terms[i][control] = ketbra[(i >> c) & 1].clone();
                c += 1;
            }
        }
        terms[n_terms - 1][target] = matrix_2x2;
        let dim = 1 << n_qubits;
        terms
            .iter()
            .fold(DMatrix::<Complex<f32>>::zeros(dim, dim), |acc1, x| {
                acc1 + x
                    .iter()
                    .fold(DMatrix::<Complex<f32>>::identity(1, 1), |acc2, y| {
                        acc2.kronecker(y)
                    })
            })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimpleError {}

#[cfg(test)]
mod tests {
    use crate::{Circuit, DebugSimulator, Instruction, SimpleSimulator};

    #[test]
    fn H04x4() {
        let mat4 = DebugSimulator::inflate_2x2(Instruction::H(0).get_matrix(), 0, 2);
        println!("{}", mat4);
        // OK
    }
    #[test]
    fn H14x4() {
        let mat4 = DebugSimulator::inflate_2x2(Instruction::H(0).get_matrix(), 1, 2);
        println!("{}", mat4);
        // OK
    }

    #[test]
    fn CNOT014x4() {
        let mat4 = DebugSimulator::controlled(Instruction::X(0).get_matrix(), vec![0], 1, 2);
        println!("{}", mat4);
        // OK
    }

    #[test]
    fn CNOT104x4() {
        let mat4 = DebugSimulator::controlled(Instruction::X(0).get_matrix(), vec![1], 0, 2);
        println!("{}", mat4);
        // OK
    }

    #[test]
    fn TOFFOLLI8x8() {
        let mat4 = DebugSimulator::controlled(Instruction::X(0).get_matrix(), vec![0, 1], 2, 3);
        println!("{}", mat4);
        // OK
    }

    #[test]
    fn foo() {
        let instructions = vec![Instruction::H(0), Instruction::CNOT(0, 1)];

        let circ = Circuit {
            instructions: instructions,
            n_qubits: 2,
        };

        let sim = DebugSimulator::build(circ).unwrap();
        for s in sim.states {
            println!("{}", s);
            // OK
        }
    }
}
