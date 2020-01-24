import typing
import unittest

import cirq
import ddt
import numpy as np
import sympy
from cirq import Rx, H, Rz, X, Z, Y, ControlledGate, XPowGate

from paulicirq.gates import TwoPauliExpGate, GlobalPhaseGate, PauliWordExpGate
# from paulicirq.gates.controlled_gates import CRz, CRx
from paulicirq.gates.gate_block import GateBlock
from paulicirq.grad import op_grad, GradNotImplemented
from paulicirq.linear_combinations import LinearCombinationOfOperations as LCO
from paulicirq.op_tree import OpTreeGenerator, VariableNQubitsGenerator
from paulicirq.pauli import PauliWord
from tests.utils import test_lco_identical_with_simulator

q0, q1, q2, q3, q4 = cirq.LineQubit.range(5)
rad = sympy.Symbol("rad")


class _2rots_op_generator(OpTreeGenerator):
    def __init__(self):
        super().__init__()
        self._num_qubits = 2

    @property
    def num_qubits(self):
        return self._num_qubits

    def params(self) -> typing.Iterable[sympy.Symbol]:
        return [rad]

    def __call__(self, qubits):
        q0, q1 = qubits[:2]
        yield Rx(rad).on(q0)
        yield Rz(rad).on(q1)


class _op_generator_3posargs_and_1varg(VariableNQubitsGenerator):
    def __call__(self, qubits):
        q0, q1 = qubits[:2]
        yield Rx(rad).on(q0)
        yield ControlledGate(Rz(rad)).on(q0, q1)
        # yield H(q2)
        # yield CNOT(q2, q1)
        # yield TOFFOLI(q1, q0, q2)

        for q in qubits:
            yield H(q)

    def params(self) -> typing.Iterable[sympy.Symbol]:
        return [rad]


u = GateBlock(
    _op_generator_3posargs_and_1varg(5)
)


class _u_positive_generator(VariableNQubitsGenerator):
    def __call__(self, qubits):
        q0, q1 = qubits[:2]
        yield Rx(rad).on(q0)
        yield ControlledGate(Z).on(q0, q1)
        yield ControlledGate(Rz(rad)).on(q0, q1)
        # yield H(q2)
        # yield CNOT(q2, q1)
        # yield TOFFOLI(q1, q0, q2)

        for q in qubits:
            yield H(q)

    def params(self) -> typing.Iterable[sympy.Symbol]:
        return [rad]


u_positive = GateBlock(_u_positive_generator(5))


class _u_negative_generator(VariableNQubitsGenerator):
    def __call__(self, qubits):
        q0, q1 = qubits[:2]
        yield Rx(rad).on(q0)
        yield ControlledGate(GlobalPhaseGate(1)).on(q0, q1)
        yield ControlledGate(Z).on(q0, q1)
        yield ControlledGate(Rz(rad)).on(q0, q1)
        # yield H(q2)
        # yield CNOT(q2, q1)
        # yield TOFFOLI(q1, q0, q2)

        for q in qubits:
            yield H(q)

    def params(self) -> typing.Iterable[sympy.Symbol]:
        return [rad]


u_negative = GateBlock(_u_negative_generator(5))


@ddt.ddt
class OperationGradTest(unittest.TestCase):

    @ddt.unpack
    @ddt.data(

        # d[Rx(rad)] / d[rad] = d[e^{-i X rad / 2}] / d[rad]
        #                     = -i/2 X e^{-i X rads / 2}
        #                     = -i/2 X Rx(rad)
        [Rx(rad).on(q0), rad,
         LCO({
             (Rx(rad).on(q0), X(q0)): -0.5j
         })],

        # d[XPow(e, s)] / d[e] = i pi {s + 1/2 (I - X)} XPow(e, s)
        [XPowGate(exponent=rad, global_shift=0.3).on(q0), rad,
         LCO({
             (XPowGate(exponent=rad, global_shift=0.3).on(q0),):
                 1.0j * np.pi * (0.3 + 0.5),
             (X(q0), XPowGate(exponent=rad, global_shift=0.3).on(q0)):
                 -0.5j * np.pi
         })],

        # d[e^{-i A0 A1 rad}] / d[rad] = -i A0 A1 e^{-i A0 A1 rad}
        [TwoPauliExpGate("X", "Z", rad).on(q0, q1), rad,
         LCO({
             (TwoPauliExpGate("X", "Z", rad).on(q0, q1), X(q0), Z(q1)): -1.0j
         })],

        # d[exp(i pi rad)] / d[rad] = i pi exp(i pi rad)
        [GlobalPhaseGate(rad ** 2).on(q0), rad,
         LCO({
             (GlobalPhaseGate(rad ** 2).on(q0),): 2.0j * sympy.pi * rad
         })],

        # d[e^{−i t P}] / d[t] = -i P e^{−i t P}
        [PauliWordExpGate(rad ** 2, PauliWord("ZYX")).on(q0, q1, q2), rad,
         LCO({
             (PauliWordExpGate(rad ** 2, PauliWord("ZYX")).on(q0, q1, q2),
              Z(q0), Y(q1), X(q2)): -2.0j * rad
         })],

        # d[Rx(rad) ⊗ Rz(rad)] / d[rad] = -i/2 { X Rx(rad) ⊗ Rz(rad)
        #                                      + Rx(rad) ⊗ Z Rz(rad) }
        [GateBlock(
            _2rots_op_generator()
        ).on(q0, q1), rad,
         -0.5j * LCO({
             (Rz(rad).on(q1), Rx(rad).on(q0), X(q0)): 1,
             (Rz(rad).on(q1), Rx(rad).on(q0), Z(q1)): 1
         })],

        # d[CRx(rad)] / d[rad] = |1><1| ⊗ d[Rx(rad)]/d[rad]
        #                      = -i/2 |1><1| ⊗ X Rx(rad)
        #                      = -i/4 { C[X Rx(rad)] - C[-X Rx(rad)] }
        [cirq.ControlledGate(Rx(rad), num_controls=1).on(q0, q1), rad,
         -1.0j / 4 * LCO({
             (ControlledGate(X).on(q0, q1), ControlledGate(Rx(rad)).on(q0, q1)): 1,
             (ControlledGate(X).on(q0, q1), ControlledGate(Rx(rad)).on(q0, q1),
              ControlledGate(GlobalPhaseGate(1)).on(q0, q1)): -1
         })],

        # d[CCRz(rad)] / d[rad] = |1><1| ⊗ |1><1| ⊗ d[Rz(rad)]/d[rad]
        #                       = -i/4 { CC[Z Rz(rad)] - CC[-Z Rz(rad)] }
        [ControlledGate(
            ControlledGate(
                cirq.Rz(rad),
                control_qubits=[q1]
            ),
            control_qubits=[q0]
        ).on(q2), rad,
         -1.0j / 4 * LCO({
             (ControlledGate(ControlledGate(Z)).on(q0, q1, q2),
              ControlledGate(ControlledGate(Rz(rad))).on(q0, q1, q2)): 1,
             (ControlledGate(ControlledGate(Z)).on(q0, q1, q2),
              ControlledGate(ControlledGate(Rz(rad))).on(q0, q1, q2),
              ControlledGate(ControlledGate(GlobalPhaseGate(1))).on(q0, q1, q2)): -1
         })],

        # d[U(rad)] = -i/2 U(rad) X.on(q0) - i/4 { U+(rad) - U-(rad) }
        [u.on(q0, q1, q2, q3, q4), rad,
         LCO({
             (X(q0), u.on(q0, q1, q2, q3, q4),): -0.5j,  # order matters!
             (u_positive.on(q0, q1, q2, q3, q4),): -0.25j,
             (u_negative.on(q0, q1, q2, q3, q4),): +0.25j
         })]

    )
    def test_grad(self, op, param, grad_lco):
        print("[Grad of {}]".format(op))
        grad = op_grad(op, param)
        if isinstance(grad, GradNotImplemented):
            print("The grad of {} is not implemented.".format(grad.operation))
            return

        for _op_series, coeff in grad.items():
            _circuit = cirq.Circuit()
            _circuit.append(_op_series)
            print("Coefficient: {}".format(coeff))
            print(_circuit, "\n")

        random_rad = typing.cast(float, np.random.rand())
        param_resolver = {"rad": random_rad}
        # param_resolver = {"rad": 0}

        test_lco_identical_with_simulator(
            cirq.resolve_parameters(grad, param_resolver),
            cirq.resolve_parameters(grad_lco, param_resolver),
            self
        )
