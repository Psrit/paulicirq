import unittest

import cirq
import numpy as np
import scipy.linalg as splinalg
import sympy
from cirq import Circuit, GridQubit, Simulator

from paulicirq.gates import PauliWordExpGate, TwoPauliExpGate, GlobalPhaseGate
from paulicirq.pauli import PauliWord, Pauli


class PauliTransformationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = Pauli.X
        self.y = Pauli.Y
        self.z = Pauli.Z
        self.i = Pauli.I
        self.paulis = [self.x, self.y, self.z]

    def test_eig_similarity(self):
        for pauli in self.paulis:
            values, vectors = np.linalg.eig(pauli)
            self.assertTrue(np.allclose(
                vectors.conjugate().T @ pauli @ vectors,
                np.diag(values)
            ))

    def test_mutually_transform(self):
        for pauli_a in self.paulis:
            for pauli_b in self.paulis:
                if not np.all(pauli_a == pauli_b):
                    values_a, vectors_a = np.linalg.eig(pauli_a)
                    values_b, vectors_b = np.linalg.eig(pauli_b)
                    self.assertTrue(np.allclose(
                        (np.linalg.inv(vectors_b.conjugate().T) @
                         vectors_a.conjugate().T @ pauli_a @ vectors_a @
                         np.linalg.inv(vectors_b)),
                        pauli_b
                    ))

    def test_exp_transform(self):
        t = np.random.rand() * 4

        exp_xz = splinalg.expm(1j * t * splinalg.kron(self.x, self.z))
        exp_zz = splinalg.expm(1j * t * splinalg.kron(self.z, self.z))
        values_x, vectors_x = np.linalg.eig(self.x)
        values_z, vectors_z = np.linalg.eig(self.z)
        self.assertTrue(np.allclose(
            (
                    splinalg.kron(np.linalg.inv(vectors_z.conjugate().T), self.i) @
                    splinalg.kron(vectors_x.conjugate().T, self.i) @
                    exp_xz @
                    splinalg.kron(vectors_x, self.i) @
                    splinalg.kron(np.linalg.inv(vectors_z), self.i)
            ),
            exp_zz
        ))


class TwoPauliExpGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit = Circuit()
        self.qubits = [GridQubit(i, j) for i in range(3) for j in range(3)]

        self.gate = TwoPauliExpGate(Pauli("X"), Pauli("Z"), 0.5)
        self.circuit.append(
            self.gate.on(self.qubits[0], self.qubits[2])
        )

    def test_diagram(self):
        self.assertEqual(
            str(self.circuit),
            "(0, 0): ───^X────────\n"
            "           │\n"
            "(0, 2): ───^Z{0.5}───"
        )

    def test_unitary(self):
        self.assertTrue(np.allclose(
            cirq.unitary(self.gate),
            np.array([
                [0.87758256 + 0.j, 0. + 0.j, 0. - 0.47942554j, 0. + 0.j],
                [0. + 0.j, 0.87758256 + 0.j, 0. + 0.j, 0. + 0.47942554j],
                [0. - 0.47942554j, 0. + 0.j, 0.87758256 + 0.j, 0. + 0.j],
                [0. + 0.j, 0. + 0.47942554j, 0. + 0.j, 0.87758256 + 0.j]
            ])
        ))

    def test_symbolic_gate(self):
        sym_circuit = Circuit()
        qubits = [GridQubit(i, j) for i in range(3) for j in range(3)]

        gate = TwoPauliExpGate(Pauli("X"), Pauli("Z"), sympy.Symbol("rad"))
        sym_circuit.append(
            gate.on(qubits[0], qubits[2])
        )

        self.assertEqual(
            str(sym_circuit),
            "(0, 0): ───^X────────\n"
            "           │\n"
            "(0, 2): ───^Z{rad}───"
        )


class PauliWordExpGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.circuit_decomposed = Circuit()
        self.circuit_compressed = Circuit()
        self.qubits = [GridQubit(i, j) for i in range(3) for j in range(3)]

        self.gate = PauliWordExpGate(0.25, PauliWord("IXXZ"))
        self.operation = self.gate.on(*self.qubits[0:4])
        self.circuit_decomposed.append(cirq.decompose(self.operation))
        self.circuit_compressed.append(self.operation)

    def test_diagram(self):
        print(self.circuit_compressed)
        print(self.circuit_decomposed)

    def test_unitary(self):
        self.assertTrue(np.allclose(
            cirq.unitary(self.gate),
            splinalg.expm(-1.0j * 0.25 * PauliWord("IXXZ").sparray.toarray())
        ))

    def test_simplest_case(self):
        circuit = Circuit()
        qubits = [GridQubit(i, j) for i in range(3) for j in range(3)]
        gate = PauliWordExpGate(0.1, PauliWord("II"))
        circuit.append(gate.on(qubits[2], qubits[0]))
        # If use the following line instead, `print(circuit)` will show nothing:
        # circuit.append(cirq.decompose(gate.on(qubits[2], qubits[0])))

        print(circuit)

        self.assertTrue(np.allclose(
            cirq.unitary(gate),
            np.identity(4)
        ))

    def test_simulate(self):
        compressed_status = [True, False]

        rad = 0.75
        pauli_word = PauliWord("XZYX")

        for initial_state in range(2 ** len(pauli_word)):
            results = []
            for is_compressed in compressed_status:
                simulator = Simulator()
                circuit = Circuit()
                qubits = [GridQubit(i, j) for i in range(3) for j in range(3)]
                gate = PauliWordExpGate(rad, pauli_word)
                operation = gate.on(*qubits[0:len(gate.pauli_word)])

                if is_compressed:
                    circuit.append(operation)
                else:
                    circuit.append(cirq.decompose(operation))

                result = simulator.simulate(
                    circuit,
                    initial_state=initial_state
                )  # type: cirq.SimulationTrialResult
                # print(circuit)
                print(result.final_simulator_state)
                # print(cirq.unitary(circuit_compressed))

                results.append(result)

            self.assertTrue(np.allclose(
                results[0].final_simulator_state.state_vector,
                results[1].final_simulator_state.state_vector
            ))

    def test_symbolic_circuit(self):
        circuit = Circuit()
        qubits = [GridQubit(i, j) for i in range(3) for j in range(3)]

        gate = PauliWordExpGate(sympy.Symbol("t"), PauliWord("XXZX"))
        operation = gate.on(qubits[0], qubits[1], qubits[3], qubits[2])
        circuit.append(cirq.decompose(operation))
        circuit.append(operation)

        print(circuit)


class GlobalPhaseGateTest(unittest.TestCase):
    def test_unitary(self):
        rad_list = np.random.rand(10) * 5
        for rad in rad_list:
            gate = GlobalPhaseGate(rad=rad)
            self.assertTrue(np.allclose(
                cirq.unitary(gate),
                np.identity(2) * np.exp(1.0j * rad * np.pi)
            ))

    def test_symbolic_gate(self):
        sym_circuit = Circuit()
        qubits = [GridQubit(i, j) for i in range(3) for j in range(3)]

        gate = GlobalPhaseGate(sympy.Symbol("rad"))
        sym_circuit.append(
            gate.on(qubits[0])
        )

        print(sym_circuit)

        rad_list = np.random.rand(10) * 5
        for rad in rad_list:
            simulator = Simulator()
            circuit = cirq.resolve_parameters(sym_circuit, {"rad": rad})
            result = simulator.simulate(circuit)
            self.assertTrue(np.allclose(
                result.final_simulator_state.state_vector,
                np.exp(1.0j * rad * np.pi) * np.array([1, 0])
            ))
