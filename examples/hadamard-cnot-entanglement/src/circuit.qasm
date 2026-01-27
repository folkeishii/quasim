OPENQASM 3;
include "stdgates.inc";

qubit[2] qs;

// Qubits are initially in an undefined state.
// Reset is used here to initialize qubit to a |0〉 state.
reset qs;

// Hadamard gate
h qs[0];

// CNOT or controlled X gate
cx qs[0], qs[1];

bit[2] my_result = measure qs;