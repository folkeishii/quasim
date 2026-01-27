use std::fs::read_to_string;
use quasim;

/* 
# Example:
Turn a |00> input into the (maximally entangled) Bell state |Φ+> = 1/√2 (|00> + |11>)

We supply a description of the circuit in OpenQASM format as well as point at which we want to inspect the state.

##  Questions:

* Would a user want to see the algebraic representation of the state at any point? 
There is an isomorphism between the state vector and algebraic representation after all.

Please use this as a starting point for discussion, and not as a final specification of the product!

*/

fn inspect_state() -> bool {
    let circuit_qasm: String = read_to_string("circuit.qasm").expect("Unable to read file");
 
    /*
    This is a 2 qubit circuit, so imagine state being a 2² component vector of some kind.
    Where something like...
    state[0] = complex value with amplitude and phase for |00>
    state[1] = complex value with amplitude and phase for |01>
    state[2] = complex value with amplitude and phase for |10>
    state[3] = complex value with amplitude and phase for |11>
    */
    let state = quasim::run_and_measure_qubits(/*circuit description:*/ &circuit_qasm, /*state inspection point:*/ "psi2");

    println!("Just for fun, here is the full state: {:?}", state); // How to print a state nicely?
    println!("And here is the measured result: {:?}", quasim::collapse_state(&state));
    
    let amplitudes = quasim::just_amplitudes(&state);
    
    // Equals operator and floating point numbers is generally a bad idea, but this is just an example!
    if amplitudes == Vec!{0.5, 0.0, 0.0, 0.5} {
        println!("The qubits are maximally entangled as expected!");
    } else {
        println!("Unexpected amplitudes: {:?}", amplitudes);
    }
}
