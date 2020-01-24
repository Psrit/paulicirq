import unittest

import ddt
from cirq import Rx, Z, CNOT

from paulicirq.linear_combinations import *

q0, q1, q2 = cirq.LineQubit.range(3)

op_tuple_1 = (Rx(sympy.Symbol("θ"))(q0), CNOT(q2, q1))
op_tuple_2 = (CNOT(q0, q2), Z(q1))

m1 = cirq.Moment(op_tuple_1)
m2 = cirq.Moment(op_tuple_2)


@ddt.ddt
class LinearSymbolicDictTest(unittest.TestCase):
    """
    A test class for `LinearSymbolicDict`, `LinearCombinationOfMoments`, and
    `LinearCombinationOfOperations` in `paulicirq.gates.linear_combinations`.

    """

    @ddt.unpack
    @ddt.data(
        [LinearCombinationOfOperations, op_tuple_1, op_tuple_2],
        [LinearCombinationOfMoments, m1, m2]
    )
    def test_numerical(self, _type, term1, term2):
        c1 = 0.123
        c2 = 0.589

        lc1 = _type({term1: c1, term2: c2})
        lc2 = _type({})

        lc2[term1] = c1
        lc2[term2] = c2

        self.assertEqual(lc1, lc2)

    @ddt.unpack
    @ddt.data(
        [LinearCombinationOfOperations, op_tuple_1, op_tuple_2],
        [LinearCombinationOfMoments, m1, m2]
    )
    def test_numerical_clean(self, _type, term1, term2):
        c1 = 0.123
        c2 = 0.589

        lc1 = _type({term1: c1, term2: c2})
        lc2 = _type({})

        lc2[term1] = -c1
        lc2[term2] = -c2

        self.assertEqual(lc1 + lc2, _type({}))

    @ddt.unpack
    @ddt.data(
        [LinearCombinationOfOperations, op_tuple_1, op_tuple_2],
        [LinearCombinationOfMoments, m1, m2]
    )
    def test_symbolic(self, _type, term1, term2):
        c1 = sympy.Symbol("c1")
        c2 = sympy.Symbol("c2")

        lc1 = _type({term1: c1, term2: c2})
        lc2 = _type({})

        lc2[term1] = c1
        lc2[term2] = c2

        self.assertEqual(lc1, lc2)

    @ddt.unpack
    @ddt.data(
        [LinearCombinationOfOperations, op_tuple_1, op_tuple_2],
        [LinearCombinationOfMoments, m1, m2]
    )
    def test_symbolic_clean(self, _type, term1, term2):
        c1 = sympy.Symbol("c1")
        c2 = sympy.Symbol("c2")

        lc1 = _type({term1: c1, term2: c2})
        lc2 = _type({})

        lc2[term1] = -c1
        lc2[term2] = -c2

        self.assertEqual(lc1 + lc2, _type({}))

    @ddt.unpack
    @ddt.data(
        [LinearCombinationOfOperations, op_tuple_1, op_tuple_2],
        [LinearCombinationOfMoments, m1, m2]
    )
    def test_is_parameterized(self, _type, term1, term2):
        c1 = sympy.Symbol("c1")
        c2 = sympy.Symbol("c2")

        lc = _type({term1: c1, term2: c2})
        self.assertTrue(cirq.is_parameterized(lc))

        lc = cirq.resolve_parameters(lc, {"c1": 0.5, "c2": 1.0})
        # print(lc)
        self.assertTrue(cirq.is_parameterized(lc))

        lc = cirq.resolve_parameters(lc, {"θ": 0.9})
        # print(lc)
        self.assertFalse(cirq.is_parameterized(lc))

    def test_simulate_lco(self):
        lco = LinearCombinationOfOperations({
            (cirq.X(q0), cirq.H(q1)): 0.5,
            (cirq.H(q0), cirq.CNOT(q0, q1)): 1.5
        })
        lco = cirq.resolve_parameters(lco, {"θ": 0.9})
        state = simulate_linear_combination_of_operations(lco, initial_state=0)

        matrix_1 = cirq.kron(cirq.unitary(cirq.X), cirq.unitary(cirq.H))
        matrix_2 = cirq.kron(cirq.unitary(cirq.H), cirq.unitary(cirq.I))
        matrix_2 = cirq.unitary(cirq.CNOT) @ matrix_2

        state_0 = (0.5 * matrix_1 + 1.5 * matrix_2) @ np.array(
            [1] + [0] * (2 ** 2 - 1)
        )
        self.assertTrue(np.allclose(state, state_0))
