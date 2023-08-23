import math
import unittest

import cirq
import numpy as np

from paulicirq.gates.hadamard_test_gate import inner_product_from_hadamard_test_result, add_hadamard_test
from paulicirq.gates.random_gates import RandomMatrixGate


class HadamardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit = cirq.Circuit()

        _qubits = cirq.LineQubit.range(10)
        self.qubits = [_qubits[0], _qubits[2], _qubits[5]]

        self._num_qubits = len(self.qubits)
        self.random_initialize = RandomMatrixGate(self._num_qubits)
        self.random_gate_a = RandomMatrixGate(self._num_qubits)
        self.random_gate_v = RandomMatrixGate(self._num_qubits)

        self.circuit.append(
            self.random_initialize.on(*self.qubits)
        )

    def test_inner_product(self):
        from copy import deepcopy

        # Calculate Re < ψ | V^\dagger A V | ψ >
        circuit_re = deepcopy(self.circuit)
        hadamard_measurement_key_re = add_hadamard_test(
            self.qubits,
            self.random_gate_v,
            self.random_gate_a,
            is_imaginary_part=False,
            circuit=circuit_re
        )
        print("Circuit for estimating the real part:\n"
              "{}\n".format(circuit_re))
        result_re = cirq.Simulator().run(circuit_re, repetitions=3000000)
        inner_product_simulator_re = \
            inner_product_from_hadamard_test_result(result_re,
                                                    hadamard_measurement_key_re,
                                                    is_imaginary_part=False)

        # Calculate Im < ψ | V^\dagger A V | ψ >
        circuit_im = deepcopy(self.circuit)
        hadamard_measurement_key_im = add_hadamard_test(
            self.qubits,
            self.random_gate_v,
            self.random_gate_a,
            is_imaginary_part=True,
            circuit=circuit_im
        )
        print("Circuit for estimating the imaginary part:\n"
              "{}\n".format(circuit_im))
        result_im = cirq.Simulator().run(circuit_im, repetitions=3000000)
        inner_product_simulator_im = \
            inner_product_from_hadamard_test_result(result_im,
                                                    hadamard_measurement_key_im,
                                                    is_imaginary_part=True)

        inner_product_simulator = (
                inner_product_simulator_re + 1.0j * inner_product_simulator_im
        )
        print(inner_product_simulator)

        # Calculate < ψ | V^\dagger A V | ψ > from the precise matrix_a forms of
        # all gates
        state_vector = cirq.kron(*[np.array([1, 0]) for _ in range(self._num_qubits)]).T
        state_vector = cirq.unitary(self.random_initialize) @ state_vector

        matrix_v = cirq.unitary(self.random_gate_v)
        matrix_a = cirq.unitary(self.random_gate_a)
        inner_product_precise = (
                state_vector.conjugate().T
                @ matrix_v.conjugate().T
                @ matrix_a
                @ matrix_v
                @ state_vector
        )

        print(inner_product_precise)

        self.assertTrue(
            math.isclose(inner_product_precise.real, inner_product_simulator.real, rel_tol=0.05)
            and
            math.isclose(inner_product_precise.imag, inner_product_simulator.imag, rel_tol=0.05)
        )
