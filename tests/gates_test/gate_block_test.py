import typing
import unittest

import cirq
import ddt
import sympy
from cirq import Rx, CNOT, H, TOFFOLI

from paulicirq.gates.gate_block import GateBlock
from paulicirq.op_tree import OpTreeGenerator, VariableNQubitsGenerator
from tests.utils import test_op_tree_identical_with_simulator


class FixedWidthGenerator3(OpTreeGenerator):
    @property
    def num_qubits(self):
        return 3

    def params(self) -> typing.Iterable[sympy.Symbol]:
        return (sympy.Symbol("rad0"),)

    def __call__(self, qubits):
        q0, q1, q2 = qubits
        yield Rx(sympy.Symbol("rad0")).on(q0)
        yield CNOT(q0, q1)
        yield H(q2)
        yield CNOT(q2, q1)
        yield TOFFOLI(q1, q0, q2)


class VariableWidthGenerator(VariableNQubitsGenerator):
    def params(self) -> typing.Iterable[sympy.Symbol]:
        return (sympy.Symbol("rad0"),)

    def __call__(self, qubits):
        q0, q1, q2 = qubits[0:3]
        yield Rx(sympy.Symbol("rad0")).on(q0)
        yield CNOT(q0, q1)
        yield H(q2)
        yield CNOT(q2, q1)
        yield TOFFOLI(q1, q0, q2)

        for q in qubits:
            yield H(q)


class FixedWidthGeneratorConstant3(OpTreeGenerator):
    @property
    def num_qubits(self):
        return 3

    def params(self) -> typing.Iterable[sympy.Symbol]:
        return ()

    def __call__(self, qubits):
        c, q0, q1 = qubits
        return cirq.decompose(cirq.CSWAP(c, q0, q1))


@ddt.ddt
class GateBlockTest(unittest.TestCase):

    @ddt.unpack
    @ddt.data(
        [FixedWidthGenerator3, 3],
        [VariableWidthGenerator, TypeError],
        {"op_generator_type": FixedWidthGeneratorConstant3,
         "val": 3}
    )
    def test_num_qubits(self, op_generator_type, val):
        if isinstance(val, type) and issubclass(val, Exception):
            with self.assertRaises(val):
                op_generator = op_generator_type()
            return

        else:
            op_generator = op_generator_type()
            block = GateBlock(op_generator)
            self.assertEqual(block.num_qubits(), val)

    @ddt.unpack
    @ddt.data(
        [FixedWidthGenerator3],
        {"op_generator_type": VariableWidthGenerator,
         "num_qubits": 5}
    )
    def test_diagram(self, op_generator_type, num_qubits=None):
        circuit = cirq.Circuit()
        if num_qubits is not None:
            op_generator = op_generator_type(num_qubits)
        else:
            op_generator = op_generator_type()
        block = GateBlock(op_generator)
        qubits = cirq.LineQubit.range(block.num_qubits())

        circuit.append(block.on(*qubits))

        print(circuit)

    def test_on(self):
        cswap_block = GateBlock(FixedWidthGeneratorConstant3())
        cswap = cirq.CSWAP
        qubits = cirq.LineQubit.range(cswap_block.num_qubits())

        test_op_tree_identical_with_simulator(
            cswap_block.on(*qubits),
            cswap.on(*qubits),
            self
        )
