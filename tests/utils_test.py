import re
import unittest

import cirq
import ddt
import numpy as np
import tensorflow as tf

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
            generate_auxiliary_qubit(
                self.circuit, auxiliary_qubit_type=cirq.LineQubit
            ).x,
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


@ddt.ddt
class DeduplicateTest(unittest.TestCase):
    @ddt.unpack
    @ddt.data(
        ((1, 2, 3, 2, 5), (1, 2, 3, 5)),
        ([1, 2, 3, 2, 5], [1, 2, 3, 5]),
    )
    def test_iterables(self, original, expected_deduplicated):
        deduplicated = deduplicate(original)
        self.assertEqual(deduplicated, expected_deduplicated)


class TFCloseTest(unittest.TestCase):
    def _test_close_helper(
        self,
        a: tf.Tensor,
        b: tf.Tensor,
        isclose: bool,
        atol=None, rtol=None
    ):
        if isclose:
            assert_func = self.assertTrue
        else:
            assert_func = self.assertFalse

        assert_func(
            tf.reduce_all(tf.equal(
                tf_isclose(a, b, atol=atol, rtol=rtol),
                tf.convert_to_tensor([True, True, True], dtype=tf.bool)
            ))
        )

        assert_func(
            tf_allclose(a, b, atol=atol, rtol=rtol)
        )

    def test_vector_close(self):
        a = tf.convert_to_tensor([1, 2, 3], dtype=tf.float16)
        b = tf.add(a, 0.001)
        atol = 1

        self._test_close_helper(a, b, isclose=True, atol=atol)

    def test_vector_not_close(self):
        a = tf.convert_to_tensor([1, 2, 3], dtype=tf.float16)
        b = tf.add(a, 0.001)
        atol = 0.0001

        self._test_close_helper(a, b, isclose=False, atol=atol)

    def test_tensor_close(self):
        a = tf.convert_to_tensor(
            [
                [[1, 2, 3],
                 [4, 5, 6]],

                [[7, 8, 9],
                 [10, 11, 12]]
            ],  # shape=(2, 2, 3)
            dtype=tf.float16)
        b = tf.add(a, 0.001)
        atol = 1

        self._test_close_helper(a, b, isclose=True, atol=atol)

    def test_tensor_not_close(self):
        a = tf.convert_to_tensor(
            [
                [[1, 2, 3],
                 [4, 5, 6]],

                [[7, 8, 9],
                 [10, 11, 12]]
            ],  # shape=(2, 2, 3)
            dtype=tf.float16
        )
        b = tf.add(a, 0.001)
        atol = 0.0001

        self._test_close_helper(a, b, isclose=False, atol=atol)


x = np.linalg.eig(cirq.unitary(cirq.X))[1]
y = np.linalg.eig(cirq.unitary(cirq.Y))[1]
z = np.linalg.eig(cirq.unitary(cirq.Z))[1]


def kron_to_tensor(state_lists):
    states = tf.convert_to_tensor(np.concatenate(
        tuple(cirq.kron(*state_list) for state_list in state_lists)
    ))
    return states


class SubstatesTest(unittest.TestCase):
    def test_direct_product(self):
        state_lists = ((x[0], y[1]),
                       (x[1], y[0]),
                       (x[0], z[1]))
        states_tensor = kron_to_tensor(state_lists)

        substates_tensor = substates(states_tensor, keep_indices=[1])
        correct_substates_tensor = tf.convert_to_tensor(
            [state_list[1] for state_list in state_lists]
        )

        inner_products_tensor = tf.abs(tf.reduce_sum(tf.multiply(
            tf.math.conj(substates_tensor), correct_substates_tensor
        ), axis=1))
        self.assertTrue(np.allclose(
            inner_products_tensor.numpy(),
            np.ones((3,))
        ))

    def test_entangled_state(self):
        entangled_state = cirq.kron(z[0], z[1]) + cirq.kron(z[1], z[0])
        entangled_state /= np.sqrt(np.conj(entangled_state) @ entangled_state.T)
        entangled_state_tensor = tf.convert_to_tensor(entangled_state)
        with self.assertRaisesRegex(
            ValueError,
            re.compile(
                "Input wavefunction could not be factored into pure state over "
                "indices \[1\]"
            )
        ):
            substates(entangled_state_tensor, keep_indices=[1])


@set_timeout(1, callback=(lambda: "Timed out."))
def tester(t):
    """Test function."""
    import time
    print("started")
    t0 = time.time()
    time.sleep(t)
    print("finished")
    t1 = time.time()

    return t1 - t0


class SetTimeoutTest(unittest.TestCase):
    def test_docstring(self):
        self.assertEqual(
            tester.__doc__,
            "Test function."
        )

    def test_didnt_timeout(self):
        import math
        t = 0.05
        delta_t = tester(t)
        self.assertTrue(
            math.isclose(t, delta_t, rel_tol=0.01, abs_tol=0.01)
        )

    def test_timed_out(self):
        t = 1.05
        timeout_info = tester(t)
        self.assertEqual(
            timeout_info,
            "Timed out."
        )
