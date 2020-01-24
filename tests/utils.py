import unittest

import cirq
import numpy as np
from cirq import OP_TREE

from paulicirq.linear_combinations import LinearCombinationOfOperations, simulate_linear_combination_of_operations
from paulicirq.gates.universal_gate_set import is_a_basic_operation


def test_gate_decomposition(
        gate: cirq.Gate,
        testcase: unittest.TestCase,
        print_circuit=True,
        expected_unitary=None
):
    qubits = cirq.LineQubit.range(2)
    control, target = qubits
    circuit_compressed = cirq.Circuit()
    circuit_decomposed = cirq.Circuit()

    circuit_compressed.append(gate.on(control, target))
    circuit_decomposed.append(
        cirq.decompose(
            gate.on(control, target),
            keep=is_a_basic_operation
        )
    )

    if print_circuit:
        print("Compressed circuit: \n{}".format(circuit_compressed))
        print("Decomposed circuit: \n{}".format(circuit_decomposed))

        print(cirq.unitary(circuit_compressed).round(3))
        print(cirq.unitary(circuit_decomposed).round(3))

    testcase.assertTrue(np.allclose(
        (cirq.unitary(circuit_compressed) if expected_unitary is None
         else expected_unitary),
        cirq.unitary(circuit_decomposed)
    ))


def test_op_tree_identical_with_simulator(
        op_tree_1: OP_TREE,
        op_tree_2: OP_TREE,
        testcase: unittest.TestCase,
):
    circuit_1 = cirq.Circuit()
    circuit_1.append(op_tree_1)

    circuit_2 = cirq.Circuit()
    circuit_2.append(op_tree_2)

    testcase.assertEqual(
        len(circuit_1.all_qubits()),
        len(circuit_2.all_qubits())
    )
    num_qubits = len(circuit_1.all_qubits())

    for i in range(0, 2 ** num_qubits):
        simulator_1 = cirq.Simulator()
        simulator_2 = cirq.Simulator()

        state_1 = (simulator_1.simulate(circuit_1, initial_state=i)
                   .final_simulator_state.state_vector)
        state_2 = (simulator_2.simulate(circuit_2, initial_state=i)
                   .final_simulator_state.state_vector)

        testcase.assertTrue(np.allclose(state_1, state_2))


def test_lco_identical_with_simulator(
        lco1: LinearCombinationOfOperations,
        lco2: LinearCombinationOfOperations,
        testcase: unittest.TestCase
):
    if cirq.is_parameterized(lco1):
        raise ValueError("Operations containing parameters are not supported! "
                         "But {} is given.".format(lco1))
    if cirq.is_parameterized(lco2):
        raise ValueError("Operations containing parameters are not supported!"
                         "But {} is given.".format(lco2))

    testcase.assertEqual(len(lco1.qubits), len(lco2.qubits))
    num_qubits = len(lco1.qubits)

    for i in range(0, 2 ** num_qubits):
        state_1 = simulate_linear_combination_of_operations(lco1, initial_state=i)
        state_2 = simulate_linear_combination_of_operations(lco2, initial_state=i)

        testcase.assertTrue(np.allclose(state_1, state_2))
