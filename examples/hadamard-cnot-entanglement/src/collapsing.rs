use std::fs::read_to_string;
use quasim;

/* 
# Example:
Turn a |00> input into the (maximally entangled) Bell state |Φ+> = 1/√2 (|00> + |11>)

We supply a description of the circuit in OpenQASM format as well as the name of the qubits to be measured.
and receive measurement results after executing the circuit.

The result here should have been collapsed to a classical bitstring of either |00> or |11> with equal probability.

All of this is built on the assumption that OpenQASM ends with a measurement of the qubits of interest. See file...

If we ever see different results, then something is wrong! Are the qubits then not maximally entangled?

A 64-bit unsigned integer is probably enough to represent the measurement result of any circuit.

##  Questions:
* What if we would like to input a different state, that cannot be represented as a classical bitstring?

* No reading of the state vector is made. If we wanted to, would we not need a new datatype? Is there some other way?

* Would a user want to see the algebraic representation of the state at any point? 
There is an isomorphism between the state vector and algebraic representation after all.

* A 64-bit unsigned integer can be the underlying representation for measured results, but maybe we should add convenience methods?

Please use this as a starting point for discussion, and not as a final specification of the product!

*/

fn simple_collapsing_test() -> bool {
    let circuit_qasm: String = read_to_string("circuit.qasm").expect("Unable to read file");

    let result = quasim::run_and_measure_qubits(/*circuit description:*/ &circuit_qasm, /*QASM variable name:*/ "my_result");

    if result == 0b00 || result == 0b11 {
        println!("Success! Measurement result: |{:02b}>", result);
        return true;
    } else {
        println!("Error! Measurement result: |{:02b}>", result);
        return false;
    }
}
