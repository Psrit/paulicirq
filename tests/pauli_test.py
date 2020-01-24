import unittest

import numpy as np

from paulicirq.pauli import Pauli, PauliWord, commutator


class PauliTest(unittest.TestCase):
    def test_pauli(self):
        for op in ["X", 0, np.array([[0, 1], [1, 0]])]:
            x = Pauli(op)
            self.assertEqual(x.string, "X")
            self.assertEqual(x.index, 0)
            self.assertTrue(np.allclose(
                x.array, np.array([[0, 1], [1, 0]])
            ))

        for op in ["I", 3, np.array([[1, 0], [0, 1]])]:
            x = Pauli(op)
            self.assertEqual(x.string, "I")
            self.assertEqual(x.index, 3)
            self.assertTrue(np.allclose(
                x.array, np.array([[1, 0], [0, 1]])
            ))

    def test_commutator(self):
        word = Pauli("Y")
        word1, word2 = word.get_the_other_two_paulis()
        self.assertTupleEqual(
            (word1.string, word2.string),
            ("Z", "X")
        )
        self.assertTrue(np.allclose(
            2j * word.array,
            commutator(word1.array, word2.array)
        ))


class PauliWordTest(unittest.TestCase):
    def setUp(self) -> None:
        self.input_labels = {
            "fullstr": "XIYXZ",
            "qubit_operator_str": "X0 Y2 X3 Z4",
            "indices": [0, 3, 1, 0, 2]
        }

        self.check_labels = {
            "fullstr": "XIYXZ",
            "qubit_operator_str": "X0 Y2 X3 Z4",
            "indices": [0, 3, 1, 0, 2]
        }

    def test_pauli_word(self):
        for _, input_v in self.input_labels.items():
            pauli_word = PauliWord(input_v)
            for check_k, check_v in self.check_labels.items():
                self.assertEqual(
                    check_v,
                    getattr(pauli_word, check_k)
                )

    def test_effective_len(self):
        self.assertEqual(
            PauliWord("XIYXZ").effective_len,
            4
        )

    def test_dict_form(self):
        self.assertEqual(
            PauliWord("XIYXZ").dict_form,
            {0: "X", 2: "Y", 3: "X", 4: "Z"}
        )

    def test_resize(self):
        word = PauliWord("XIYXZ")

        self.assertEqual(
            PauliWord(word, length=3),
            PauliWord("XIY")
        )
        self.assertEqual(
            PauliWord(word.fullstr, length=3),
            PauliWord("XIY")
        )
        self.assertEqual(
            PauliWord(word.qubit_operator_str, length=3),
            PauliWord("XIY")
        )

        self.assertEqual(
            PauliWord(word, length=8),
            PauliWord("XIYXZIII")
        )
        self.assertEqual(
            PauliWord(word.fullstr, length=8),
            PauliWord("XIYXZIII")
        )
        self.assertEqual(
            PauliWord(word.qubit_operator_str, length=8),
            PauliWord("XIYXZIII")
        )

    def test_array(self):
        self.assertTrue(np.allclose(
            PauliWord("XIY").sparray.toarray(),
            np.array([
                [0, 0, 0, 0, 0, -1j, 0, 0],
                [0, 0, 0, 0, 1j, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1j],
                [0, 0, 0, 0, 0, 0, 1j, 0],
                [0, -1j, 0, 0, 0, 0, 0, 0],
                [1j, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0, 0, 0],
                [0, 0, 1j, 0, 0, 0, 0, 0]
            ])
        ))

    def test_pauli_word_slice(self):
        self.assertEqual(
            PauliWord("XIYXZ")[1:5:2],
            PauliWord("IX")
        )

        self.assertTrue(
            PauliWord("XIYXZ")[-1],
            PauliWord("Z")
        )

    def test_identity(self):
        self.assertEqual(
            PauliWord("I" * 10).qubit_operator_str,
            ""
        )
