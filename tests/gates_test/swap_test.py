import math
import unittest

import cirq
import numpy as np

from paulicirq.gates.swap_test_gate import add_swap_test


class SWAPTestGateTestOnTwoQubits(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit = cirq.Circuit()
        self.qubits = cirq.LineQubit.range(2)
        self.simulator = cirq.Simulator()

    def test_equal(self):
        self.circuit.append([
            cirq.X(self.qubits[0]),
            cirq.X(self.qubits[1]),
        ])

        measurement = add_swap_test(
            self.qubits[0],
            self.qubits[1],
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=10)
        print(result)
        self.assertTrue(np.all(result.measurements[measurement] == 0))

    def test_unequal(self):
        self.circuit.append([
            cirq.X(self.qubits[0]),

            cirq.X(self.qubits[1]),
            cirq.H(self.qubits[1])
        ])

        measurement = add_swap_test(
            self.qubits[0],
            self.qubits[1],
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=10)
        print(result)
        self.assertTrue(np.any(result.measurements[measurement] == 1))

    def test_orthogonal(self):
        self.circuit.append([
            cirq.X(self.qubits[0]),
        ])

        measurement = add_swap_test(
            self.qubits[0],
            self.qubits[1],
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=2000)
        count = result.histogram(key=measurement)
        print(count)
        self.assertTrue(math.isclose(count[0], count[1], rel_tol=0.10))


class SWAPTestGateTestOnTwoStates(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit = cirq.Circuit()
        self.state1 = cirq.LineQubit.range(0, 3)
        self.state2 = cirq.LineQubit.range(3, 6)
        self.simulator = cirq.Simulator()

    def test_equal(self):
        """
        For two states |ψ> and |φ>, as long as they are equal, the outcome of
        the SWAP test measurement (O) MUST always be 0, i.e.:

        |ψ> == |φ> ==> (O == 0) == True

        """
        for qubit1, qubit2 in zip(self.state1, self.state2):
            self.circuit.append([
                cirq.X(qubit1),
                cirq.X(qubit2),
            ])

        measurement = add_swap_test(
            self.state1,
            self.state2,
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=10)
        print(result)
        self.assertTrue(np.all(result.measurements[measurement] == 0))

    def test_unequal(self):
        """
        For two states |ψ> and |φ>, as long as they are not equal, the outcomes
        of the SWAP test measurements (O) MUST always contain 1 if we have large
        enough number of repeats of the SWAP test, i.e.:

        |ψ> != |φ> ==> (repeats of O contain 1) == True

        """
        for qubit1, qubit2 in zip(self.state1, self.state2):
            self.circuit.append([
                cirq.X(qubit1),

                cirq.X(qubit2),
                cirq.H(qubit2)
            ])

        measurement = add_swap_test(
            self.state1,
            self.state2,
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=10)
        print(result)
        self.assertTrue(np.any(result.measurements[measurement] == 1))

    def test_orthogonal(self):
        for qubit1, qubit2 in zip(self.state1, self.state2):
            self.circuit.append([
                cirq.X(qubit1),
            ])

        measurement = add_swap_test(
            self.state1,
            self.state2,
            self.circuit
        )

        print(self.circuit)

        result = self.simulator.run(self.circuit, repetitions=2000)
        count = result.histogram(key=measurement)
        print(count)
        self.assertTrue(math.isclose(count[0], count[1], rel_tol=0.10))

    def test_inner_product(self):
        from tests.gates_test._inner_product_test_utils import _test_inner_product
        inner_product_simulator, inner_product_exact = _test_inner_product("overlap")
        print(inner_product_simulator, inner_product_exact)

        self.assertTrue(
            math.isclose(inner_product_simulator, abs(inner_product_exact),
                         rel_tol=0.20)
        )
