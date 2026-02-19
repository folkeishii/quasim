OPENQASM 3;
include "stdgates.inc";

qubit[2] qs;

// Qubits are initially in an undefined state.
// Reset is used here to initialize qubit to a |0〉 state.
reset qs;

//psi0

// Hadamard gate
h qs[0];

//psi1

// CNOT or controlled X gate
cx qs[0], qs[1];

//psi2

bit[2] my_result = measure qs;

// There is a syntax for pragmas in OpenQASM but other compilers will refuse to try this file if we invent our own.
// Comments may therefore be an option for keeping code compatibility with other quantum simulators. 