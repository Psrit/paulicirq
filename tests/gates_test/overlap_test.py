import math
import unittest
from collections import Counter

import cirq
import numpy as np

from paulicirq.gates.overlap_test_gate import add_overlap_test


class OverlapTestGateTestOnTwoQubits(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit = cirq.Circuit()
        self.qubits = cirq.LineQubit.range(2)
        self.simulator = cirq.Simulator()

    def test_equal(self):
        """
        For two states |ψ> and |φ>, as long as they are equal, the two outcomes
        of the overlap measurement (O1 and O2) cannot both be 1 at the same time,
        i.e.:

        (O1 == 1 and O2 == 1) == False

        which is equivalent to:

        O1 == 0 or O2 == 0

        """
        self.circuit.append([
            cirq.X(self.qubits[0]),
            cirq.X(self.qubits[1]),
        ])

        measurement = add_overlap_test(
            self.qubits[0],
            self.qubits[1],
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=10)
        print(result.measurements)
        postprocessed_outcome = np.any(
            result.measurements[measurement + " - pair 0"] == 0, axis=1
        )
        self.assertTrue(
            np.all(postprocessed_outcome)
        )

    def test_unequal(self):
        """
        For two states |ψ> and |φ>, as long as they are not equal, if we have
        large enough number of repeats of overlap tests, there MUST be a test
        case where the two outcomes of the overlap measurement (O1 and O2) can
        both be 1 at the same time.

        """
        self.circuit.append([
            cirq.X(self.qubits[0]),

            cirq.X(self.qubits[1]),
            cirq.H(self.qubits[1])
        ])

        measurement = add_overlap_test(
            self.qubits[0],
            self.qubits[1],
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=10)
        print(result)
        postprocessed_outcome = np.logical_and(
            result.measurements[measurement + " - pair 0"][:, 0],
            result.measurements[measurement + " - pair 0"][:, 1]
        )
        self.assertTrue(
            np.any(postprocessed_outcome)
        )

    def test_orthogonal(self):
        self.circuit.append([
            cirq.X(self.qubits[0]),
        ])

        measurement = add_overlap_test(
            self.qubits[0],
            self.qubits[1],
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=2000)
        print(result)
        postprocessed_outcome = np.logical_and(
            result.measurements[measurement + " - pair 0"][:, 0],
            result.measurements[measurement + " - pair 0"][:, 1]
        )
        count = Counter(postprocessed_outcome)
        print(count)
        self.assertTrue(math.isclose(count[0], count[1], rel_tol=0.10))


class OverlapTestGateTestOnTwoStates(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit = cirq.Circuit()
        self.state1 = cirq.LineQubit.range(0, 3)
        self.state2 = cirq.LineQubit.range(3, 6)
        self.simulator = cirq.Simulator()

    def test_equal(self):
        """
        For two states |ψ> and |φ>, as long as they are equal, for any pair of
        all the outcomes of the overlap measurements (e.g. Oi1 and Oi2 for the
        i-th pair outcome in a certain overlap test) cannot both be 1 at the
        same time, i.e.:

        (Oi1 == 1 and Oi2 == 1) == False

        which is equivalent to:

        Oi1 == 0 or Oi2 == 0

        """
        for qubit1, qubit2 in zip(self.state1, self.state2):
            self.circuit.append([
                cirq.X(qubit1),
                cirq.X(qubit2),
            ])

        measurement = add_overlap_test(
            self.state1,
            self.state2,
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=10)
        print(result.measurements)

        overlap_test_result = []
        for mkey, mresult in result.measurements.items():
            if mkey.startswith(measurement):
                postprocessed_outcome = np.any(
                    mresult == 0, axis=1
                )
                overlap_test_result.append(postprocessed_outcome)

        print(overlap_test_result)
        print(len(overlap_test_result))

        self.assertTrue(
            np.all(overlap_test_result)
        )

    def test_unequal(self):
        """
        For two states |ψ> and |φ>, as long as they are not equal, if we have
        large enough number of repeats of overlap tests, there MUST be a test
        case which "fail"ed.

        """
        for qubit1, qubit2 in zip(self.state1, self.state2):
            self.circuit.append([
                cirq.X(qubit1),

                cirq.X(qubit2),
                cirq.H(qubit2)
            ])

        measurement = add_overlap_test(
            self.state1,
            self.state2,
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=10)
        print(result.measurements)

        overlap_test_result = 0.0
        for mkey, mresult in result.measurements.items():
            if mkey.startswith(measurement):
                postprocessed_outcome = np.logical_and(
                    mresult[:, 0],
                    mresult[:, 1]
                )
                overlap_test_result += postprocessed_outcome

        counter = Counter(overlap_test_result % 2)
        print(counter)

        self.assertTrue(counter[1] != 0)  # the number of "fail"s in the SWAP test

    def test_inner_product(self):
        from tests.gates_test._inner_product_test_utils import _test_inner_product
        inner_product_simulator, inner_product_exact = _test_inner_product("overlap")
        print(inner_product_simulator, inner_product_exact)

        self.assertTrue(
            math.isclose(inner_product_simulator, abs(inner_product_exact),
                         rel_tol=0.20)
        )
