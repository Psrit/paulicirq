import unittest

from paulicirq.pauli import PauliWord
from paulicirq.utils import *


class GetAllLineQubitIdsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit = cirq.Circuit()

        self.line_qubits = cirq.LineQubit.range(100)
        self.grid_qubits = [
            cirq.GridQubit(i, j) for i in range(10) for j in range(10)
        ]

        self.circuit.append([
            cirq.X(self.line_qubits[15]),
            cirq.CZ(self.line_qubits[99], self.grid_qubits[52]),
            cirq.H(self.line_qubits[20])
        ])

    def test_get_all_line_qubit_ids(self):
        self.assertEqual(
            get_all_line_qubit_ids(self.circuit),
            (15, 20, 99)
        )

    def test_generate_auxiliary_qubit(self):
        self.assertEqual(
            generate_auxiliary_qubit(self.circuit).x,
            9999
        )


class PauliExpansionTest(unittest.TestCase):
    def test_for_random_matrix(self):
        num_qubits = 2 ** 3

        matrix = random_complex_matrix(num_qubits, num_qubits)

        expansion = pauli_expansion_for_any_matrix(matrix)

        _matrix = 0.0
        for term, coeff in expansion.items():
            _matrix += coeff * PauliWord(term).sparray.toarray()

        self.assertTrue(np.allclose(_matrix, matrix))
