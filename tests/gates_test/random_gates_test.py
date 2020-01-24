import unittest

import cirq
import numpy as np

from paulicirq.gates.random_gates import RandomMatrixGate


class RandomMatrixGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit = cirq.Circuit()
        self.qubits = cirq.LineQubit.range(10)
        self.random_gate = RandomMatrixGate(num_qubits=3)

    def test_incorrect_qubit_number(self):
        with self.assertRaises(ValueError):
            self.circuit.append(self.random_gate.on(self.qubits[0], self.qubits[3]))

    def test_diagram(self):
        self.circuit.append(self.random_gate.on(self.qubits[0], self.qubits[2], self.qubits[5]))

        print(self.circuit)

    def test_unitary(self):
        matrix = cirq.unitary(self.random_gate)

        self.assertTrue(np.allclose(
            matrix @ matrix.conjugate().T,
            np.identity(2 ** self.random_gate.num_qubits())
        ))
