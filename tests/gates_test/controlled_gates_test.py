import unittest

import cirq
import numpy as np
import sympy

from paulicirq.gates.controlled_gates import (
    Controlled1BitMatrixGate, CRx, CRy, CRz, ControlledEigenGate
)
from paulicirq.gates.random_gates import RandomMatrixGate
from paulicirq.gates.universal_gate_set import is_a_basic_operation
from tests.utils import test_gate_decomposition


class Controlled1BitMatrixGateTest(unittest.TestCase):
    def test_identity(self):
        identity = cirq.I
        c1u1 = Controlled1BitMatrixGate(sub_gate=identity)
        test_gate_decomposition(
            c1u1, self, expected_unitary=np.array([[1.0]]),
            print_circuit=False
        )

    def test_pauli_x(self):
        pauli_x = cirq.X
        c1u1 = Controlled1BitMatrixGate(sub_gate=pauli_x)
        test_gate_decomposition(c1u1, self, print_circuit=False)

    def test_pauli_y(self):
        pauli_y = cirq.Y
        c1u1 = Controlled1BitMatrixGate(sub_gate=pauli_y)
        test_gate_decomposition(c1u1, self, print_circuit=False)

    def test_pauli_z(self):
        pauli_z = cirq.Z
        c1u1 = Controlled1BitMatrixGate(sub_gate=pauli_z)
        test_gate_decomposition(c1u1, self, print_circuit=False)

    def test_random_matrix_gate(self):
        sub_gate = RandomMatrixGate(num_qubits=1)
        c1u1 = Controlled1BitMatrixGate(sub_gate=sub_gate)
        test_gate_decomposition(c1u1, self, print_circuit=False)


class CRotGateTest(unittest.TestCase):
    NUM_TESTS = 100

    def test_symbolic_c_rx(self):
        gate = CRx(sympy.Symbol("θ"))

        qubits = cirq.LineQubit.range(2)
        control, target = qubits
        circuit_decomposed = cirq.Circuit()

        circuit_decomposed.append(
            cirq.decompose(
                gate.on(control, target),
                keep=is_a_basic_operation
            )
        )

        # print("Decomposed circuit of Rx(θ): \n{}".format(circuit_decomposed))
        self.assertEqual(
            str(circuit_decomposed),
            "0: ──────────────@────────────────@───────────────────────────\n"
            "                 │                │\n"
            "1: ───Rz(0.5π)───X───Ry(-0.5*θ)───X───Ry(0.5*θ)───Rz(-0.5π)───"
        )

    def test_symbolic_c_ry(self):
        gate = CRy(sympy.Symbol("θ"))

        qubits = cirq.LineQubit.range(2)
        control, target = qubits
        circuit_decomposed = cirq.Circuit()

        circuit_decomposed.append(
            cirq.decompose(
                gate.on(control, target),
                keep=is_a_basic_operation
            )
        )

        # print("Decomposed circuit of Ry(θ): \n{}".format(circuit_decomposed))
        self.assertEqual(
            str(circuit_decomposed),
            "0: ───@────────────────@───────────────\n"
            "      │                │\n"
            "1: ───X───Ry(-0.5*θ)───X───Ry(0.5*θ)───"
        )

    def test_symbolic_c_rz(self):
        gate = CRz(sympy.Symbol("θ"))

        qubits = cirq.LineQubit.range(2)
        control, target = qubits
        circuit_decomposed = cirq.Circuit()

        circuit_decomposed.append(
            cirq.decompose(
                gate.on(control, target),
                keep=is_a_basic_operation
            )
        )

        # print("Decomposed circuit of Rz(θ): \n{}".format(circuit_decomposed))
        self.assertEqual(
            str(circuit_decomposed),
            "0: ───────────────@────────────────@───\n"
            "                  │                │\n"
            "1: ───Rz(0.5*θ)───X───Rz(-0.5*θ)───X───"
        )

    def test_c_rx(self):
        for i in range(self.NUM_TESTS):
            rads = np.random.rand() * 2 * np.pi  # [0, 2π)
            c_rx = CRx(rads)
            test_gate_decomposition(c_rx, self, print_circuit=False)

    def test_c_ry(self):
        for i in range(self.NUM_TESTS):
            rads = np.random.rand() * 2 * np.pi  # [0, 2π)
            c_ry = CRy(rads)
            test_gate_decomposition(c_ry, self, print_circuit=False)

    def test_c_rz(self):
        for i in range(self.NUM_TESTS):
            rads = np.random.rand() * 2 * np.pi  # [0, 2π)
            c_rz = CRz(rads)
            test_gate_decomposition(c_rz, self, print_circuit=False)


class ControlledEigenGateTest(unittest.TestCase):
    def test_mro(self):
        self.assertEqual(
            ControlledEigenGate.__mro__,
            (ControlledEigenGate, cirq.ControlledGate, cirq.EigenGate, cirq.Gate, object)
        )

    def test_eigen_components(self):
        cq0, cq1, q0 = cirq.LineQubit.range(3)
        ccrz = ControlledEigenGate(
            ControlledEigenGate(
                cirq.rz(0.123) ** 0.956,
            ),
        )

        circuit = cirq.Circuit()
        circuit.append(ccrz.on(cq0, cq1, q0))

        eigen_components = dict(ccrz._eigen_components())
        eigen_components_0 = {
            0: np.diag([1] * (2 ** 3 - 2 ** 1) + [0] * (2 ** 1)),
            0.5: np.diag([0] * (2 ** 3 - 2 ** 1) + [0, 1]),
            -0.5: np.diag([0] * (2 ** 3 - 2 ** 1) + [1, 0])
        }

        self.assertEqual(
            eigen_components.keys(), eigen_components_0.keys()
        )
        for k in eigen_components.keys():
            self.assertTrue(np.all(
                eigen_components[k] == eigen_components_0[k]
            ))

        e = 0.123 / np.pi * 0.956
        self.assertEqual(ccrz.exponent, e)
        self.assertEqual(ccrz._global_shift, 0)

        self.assertEqual(
            str(circuit),
            "0: ───@────────────\n"
            "      │\n"
            "1: ───@────────────\n"
            "      │\n"
            "2: ───Rz(0.037π)───"
        )

    def test_unitary(self):
        crz = ControlledEigenGate(
            cirq.rz(0.123)
        )

        s = crz._global_shift
        e = crz._exponent
        # print(crz.sub_gate_exponent)
        # print(s, e, crz._eigen_components())
        unitary_from_eigen_components = sum(
            np.exp(1.0j * np.pi * (theta + s) * e) * projector
            for theta, projector in crz._eigen_components()
        )

        # print(cirq.unitary(crz))
        # print(unitary_from_eigen_components)
        self.assertTrue(np.allclose(
            cirq.unitary(crz),
            unitary_from_eigen_components
        ))
