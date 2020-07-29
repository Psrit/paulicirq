import unittest

import numpy as np
import openfermion

from paulicirq.algorithms.qubit_reduce import (
    reduce_inactive_qubits,
    inactivate_stationary_qubits,
    reduce,
    group,
    block_reduce
)


class TestQubitReduction(unittest.TestCase):
    def setUp(self) -> None:
        self.a, self.b, self.c, self.d = np.random.rand(4)
        self.qubit_op = (
                openfermion.QubitOperator("X0 Y1 Z2    X5   ", self.a) +
                openfermion.QubitOperator("   Y1    Y3 X5   ", self.b) +
                openfermion.QubitOperator("Z0 Y1 X2 X3 X5   ", self.c) +
                openfermion.QubitOperator("   Y1       X5 Y7", self.d)
        )
        self.inactive_qubits_reduced_qubit_op, _ = \
            reduce_inactive_qubits(self.qubit_op)
        self.stationary_qubits_inactivated_qubit_op, _ = \
            inactivate_stationary_qubits(self.qubit_op)
        self.reduced_qubit_op, _, _ = reduce(self.qubit_op)

    def test_reduce_inactive_qubits(self):
        self.assertEqual(
            self.inactive_qubits_reduced_qubit_op,
            openfermion.QubitOperator("X0 Y1 Z2 X4", self.a)
            + openfermion.QubitOperator("Y1 Y3 X4", self.b)
            + openfermion.QubitOperator("Z0 Y1 X2 X3 X4", self.c)
            + openfermion.QubitOperator("Y1 X4 Y5", self.d)
        )

    def test_reduce_stationary_qubits(self):
        self.assertEqual(
            self.stationary_qubits_inactivated_qubit_op,
            openfermion.QubitOperator("X0 Z2", self.a)
            + openfermion.QubitOperator("Y3", self.b)
            + openfermion.QubitOperator("Z0 X2 X3", self.c)
            + openfermion.QubitOperator("Y7", self.d)
        )

    def test_reduce(self):
        self.assertEqual(
            self.reduced_qubit_op,
            openfermion.QubitOperator("X0 Z1", self.a)
            + openfermion.QubitOperator("Y2", self.b)
            + openfermion.QubitOperator("Z0 X1 X2", self.c)
            + openfermion.QubitOperator("Y3", self.d)
        )

    def test_group(self):
        grouped = group(self.qubit_op, by_qubit_indices=(3, 5))
        self.assertEqual(
            grouped,
            {
                ((3, "I"), (5, "X")):
                    openfermion.QubitOperator("X0 Y1 Z2    X5   ", self.a) +
                    openfermion.QubitOperator("   Y1       X5 Y7", self.d),
                ((3, "Y"), (5, "X")):
                    openfermion.QubitOperator("   Y1    Y3 X5   ", self.b),
                ((3, "X"), (5, "X")):
                    openfermion.QubitOperator("Z0 Y1 X2 X3 X5   ", self.c)
            }
        )

    def test_eigvalsh(self):
        a, b, c, d, e = 1, 2, 3, 4, 5
        qubit_op = (
                openfermion.QubitOperator("X0 Y1 Z2", a)
                + openfermion.QubitOperator("Y1 X2", b)
                + openfermion.QubitOperator("Y1 X4", c)
                + openfermion.QubitOperator("Z0 Y1 X2", d)
                + openfermion.QubitOperator("Z0 Y1 Z4", e)
        )

        inactive_qubits_reduced_qubit_op, _ = \
            reduce_inactive_qubits(qubit_op)
        stationary_qubits_inactivated_qubit_op, _ = \
            inactivate_stationary_qubits(qubit_op)

        reduced_qubit_op, n_stationary, n_inactive = reduce(qubit_op)

        # # n_stationary==1, n_inactive==1 in this case
        # self.assertEqual(n_stationary, 1)
        # self.assertEqual(n_inactive, 1)

        # e.g.
        # [-11,    -11,    -11,    -11,
        #   -3,     -3,     -3,     -3,
        #    3,      3,      3,      3,
        #   11,     11,     11,     11]
        eig = np.linalg.eigvalsh(
            openfermion.get_sparse_operator(qubit_op)
            .toarray()
        )

        # e.g.
        # [-11,    -11,
        #   -3,     -3,
        #    3,      3,
        #   11,     11]
        eig_inactive_qubits_reduced = np.linalg.eigvalsh(
            openfermion.get_sparse_operator(inactive_qubits_reduced_qubit_op)
            .toarray()
        )

        # Note that `inactivate_stationary_qubits` won't reduce the dimension of
        # the multi-qubit space of the qubit operator.
        eig_stationary_qubits_inactivated = np.linalg.eigvalsh(
            openfermion.get_sparse_operator(stationary_qubits_inactivated_qubit_op)
            .toarray()
        )

        # e.g.
        # [-11,
        #   -3,
        #    3,
        #   11]
        eig_reduced = np.linalg.eigvalsh(
            openfermion.get_sparse_operator(reduced_qubit_op)
            .toarray()
        )

        self.assertTrue(np.allclose(
            eig,
            np.repeat(eig_inactive_qubits_reduced, 2 ** n_inactive)
        ))
        self.assertTrue(np.allclose(
            eig,
            np.repeat(eig_reduced, 2 ** (n_stationary + n_inactive))
        ))

    def test_nonzero_op_constant(self):
        # matrix_a form:
        # [[ 0.96,  0.   ]
        #  [ 0.  , -0.49 ]]
        # qubit_op.constant == 0.235
        qubit_op = (
                openfermion.QubitOperator("", 0.235)
                + openfermion.QubitOperator("Z0", 0.725)
        )

        self.assertEqual(
            reduce_inactive_qubits(qubit_op),
            (qubit_op, 0)
        )

        self.assertEqual(
            inactivate_stationary_qubits(qubit_op),
            (qubit_op, 0)
        )

        self.assertEqual(
            reduce(qubit_op),
            (qubit_op, 0, 0)
        )

    def test_block_reduce(self):
        const = 1
        qubit_op = (
                openfermion.QubitOperator("", const) +
                openfermion.QubitOperator("X0 Y1 Z2    X5    X8", self.a) +
                openfermion.QubitOperator("   Y1    Y3 X5    X8", self.b) +
                openfermion.QubitOperator("Z0 Y1 X2 X3 X5 Y7 X8", self.c) +
                openfermion.QubitOperator("   Y1       X5 Y7   ", self.d)
        )
        self.assertEqual(
            block_reduce(qubit_op),
            {
                ((1, "I"), (4, "I"), (5, "I"), (6, "I"), (7, "I"), (8, "I")):
                    openfermion.QubitOperator("", const),
                ((1, "Y"), (4, "I"), (5, "X"), (6, "I"), (7, "I"), (8, "X")):
                    openfermion.QubitOperator("X0 Z1   ", self.a) +
                    openfermion.QubitOperator("      Y2", self.b),
                ((1, "Y"), (4, "I"), (5, "X"), (6, "I"), (7, "Y"), (8, "X")):
                    openfermion.QubitOperator("Z0 X1 X2", self.c),
                ((1, "Y"), (4, "I"), (5, "X"), (6, "I"), (7, "Y"), (8, "I")):
                    openfermion.QubitOperator("", self.d)
            }
        )
