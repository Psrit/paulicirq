import unittest

import cirq
from scipy.sparse import linalg as splinalg

from paulicirq.algorithms.pauliword import pauli_word_exp_factorization
from paulicirq.pauli import PauliWord


class TwoQubitMatrixGateTest(unittest.TestCase):
    def test_diagram(self):
        qubits = [cirq.GridQubit(i, j) for i in range(2) for j in range(2)]
        circuit = cirq.circuits.Circuit()

        circuit.append(
            cirq.MatrixGate(
                splinalg.expm(1j * 0.1 * PauliWord("XX").sparray.toarray())
            )(*qubits[0:2])
        )

        print(circuit)
        self.assertEqual(
            str(circuit),
            "           ┌                                           ┐\n"
            "           │0.995+0.j  0.   +0.j  0.   +0.j  0.   +0.1j│\n"
            "(0, 0): ───│0.   +0.j  0.995+0.j  0.   +0.1j 0.   +0.j │───\n"
            "           │0.   +0.j  0.   +0.1j 0.995+0.j  0.   +0.j │\n"
            "           │0.   +0.1j 0.   +0.j  0.   +0.j  0.995+0.j │\n"
            "           └                                           ┘\n"
            "           │\n"
            "(0, 1): ───#2──────────────────────────────────────────────"
        )


class PauliWordExpFactorizationTest(unittest.TestCase):
    def test_factorization(self):
        factorization = pauli_word_exp_factorization(0.1, PauliWord("XIXXYZ"))
        self.assertEqual(
            factorization,
            [(-0.7853981633974483, PauliWord("IIIIXZ")),
             (-0.7853981633974483, PauliWord("IIIZZI")),
             (0.7853981633974483, PauliWord("IIIIXZ")),
             (-0.7853981633974483, PauliWord("IIZYII")),
             (0.1, PauliWord("XIYIII")),
             (0.7853981633974483, PauliWord("IIZYII")),
             (-0.7853981633974483, PauliWord("IIIIXZ")),
             (0.7853981633974483, PauliWord("IIIZZI")),
             (0.7853981633974483, PauliWord("IIIIXZ"))]
        )

    def test_symbolic_factorization(self):
        import sympy
        factorization = pauli_word_exp_factorization(sympy.Symbol("t"), PauliWord("XIXXYZ"))
        self.assertEqual(
            factorization,
            [(-0.7853981633974483, PauliWord("IIIIXZ")),
             (-0.7853981633974483, PauliWord("IIIZZI")),
             (0.7853981633974483, PauliWord("IIIIXZ")),
             (-0.7853981633974483, PauliWord("IIZYII")),
             (sympy.Symbol("t"), PauliWord("XIYIII")),
             (0.7853981633974483, PauliWord("IIZYII")),
             (-0.7853981633974483, PauliWord("IIIIXZ")),
             (0.7853981633974483, PauliWord("IIIZZI")),
             (0.7853981633974483, PauliWord("IIIIXZ"))]
        )

    def test_factorization_special_case(self):
        """
        This test case reveals the bug of `pauli_word_exp_factorization` when
        choosing `k`: use `sorted` instead of `list` to create list from
        `pauli_dict_form.keys()` (which is actually unordered).
        """
        factorization = pauli_word_exp_factorization(1.2, PauliWord("ZZIIIIIZZ"))
        print(factorization)
        self.assertEqual(
            len(factorization),
            5
        )
