import typing
from typing import List, Union, Tuple

import cirq
import numpy as np
import sympy
from cirq.ops.eigen_gate import EigenComponent

from paulicirq.utils import is_complex_close
from .pauli_exp_gates import GlobalPhaseGate


class Controlled1BitMatrixGate(cirq.ControlledGate):
    def __init__(
            self,
            sub_gate: typing.Union[
                cirq.MatrixGate, np.ndarray
            ]
    ):
        if isinstance(sub_gate, np.ndarray):
            sub_gate = cirq.MatrixGate(sub_gate, qid_shape=(2,))

        if not cirq.has_unitary(sub_gate):
            raise ValueError("Cannot get unitary from {}".format(sub_gate))

        super(Controlled1BitMatrixGate, self).__init__(
            sub_gate, num_controls=1
        )

    def _decompose_(self, qubits):
        control, target = qubits

        if np.all(cirq.unitary(self.sub_gate) == cirq.unitary(cirq.X)):
            yield cirq.CNOT(control=control, target=target)
            return

        elif np.all(cirq.unitary(self.sub_gate) == cirq.unitary(cirq.I)):
            return

        _matrix = cirq.unitary(self.sub_gate)
        [a, b], [c, d] = _matrix

        alpha = beta = delta = gamma = 0.0

        if b == c == 0.0:
            gamma = delta = 0.0
            alpha = np.angle(a * d) / 2.0
            beta = np.angle(d / a)

        elif a == d == 0.0:
            gamma = np.pi
            delta = 0.0
            alpha = np.angle(- b * c) / 2.0
            beta = np.angle(-c / b)

        else:  # a * b * c * d != 0.0
            alpha = np.angle(- b * c) / 2.0
            beta = -np.angle(a / c)
            gamma = 2 * np.arccos(np.abs(a))
            delta = -np.angle(-a / b)

        if a != 0.0:
            a_evaluated = (np.exp(1.0j * (alpha - beta / 2.0 - delta / 2.0))
                           * np.cos(gamma / 2.0))
            # if there is a difference in the complex angle
            if is_complex_close(a, -a_evaluated,
                                atol=1e-10):  # atol is required for near zero values
                alpha = alpha + np.pi if alpha < 0 else alpha - np.pi
            # else:
            #     assert is_complex_close(a, a_evaluated, atol=1e-9)

        else:
            b_evaluated = (-np.exp(1.0j * (alpha - beta / 2.0 + delta / 2.0))
                           * np.sin(gamma / 2.0))
            # if there is a difference in the complex angle
            if is_complex_close(b, -b_evaluated,
                                atol=1e-10):
                alpha = alpha + np.pi if alpha < 0 else alpha - np.pi
            # else:
            #     assert is_complex_close(b, b_evaluated, atol=1e-9)

        # Gate C
        yield cirq.rz((delta - beta) / 2.0).on(target)

        # Gate CNOT
        yield cirq.CNOT(control=control, target=target)

        # Gate B
        yield cirq.rz(-(delta + beta) / 2.0).on(target)
        yield cirq.ry(-gamma / 2.0).on(target)

        # Gate CNOT
        yield cirq.CNOT(control=control, target=target)

        # Gate A
        yield cirq.ry(gamma / 2.0).on(target)
        yield cirq.rz(beta).on(target)

        # Gate Phase
        yield cirq.rz(alpha).on(control)
        yield GlobalPhaseGate(alpha / (2.0 * np.pi)).on(control)

    def on(self, control, target):
        return cirq.GateOperation(self, [control, target])


@cirq.value.value_equality(distinct_child_types=True, approximate=True)
class ControlledEigenGate(cirq.ControlledGate, cirq.EigenGate):
    def __init__(
            self,
            sub_gate: cirq.EigenGate,
            num_controls: int = None,
            global_shift: float = 0.0,
            exponent: float = 1.0
    ):
        if not isinstance(sub_gate, cirq.EigenGate):
            raise ValueError("An EigenGate is required.")

        super(ControlledEigenGate, self).__init__(
            sub_gate=sub_gate,
            num_controls=num_controls
        )  # Equiv. to ControlledGate.__init__(...)

        self._global_shift = global_shift
        self._exponent = self.sub_gate_exponent * exponent
        self._canonical_exponent_cached = None

    @property
    def sub_gate_exponent(self):
        return self.sub_gate._exponent

    @property
    def sub_gate_global_shift(self):
        return self.sub_gate._global_shift

    def _eigen_components(self) -> List[Union[EigenComponent,
                                              Tuple[float, np.ndarray]]]:
        _sub_eig_comp = self.sub_gate._eigen_components()
        sub_gate_shape = pow(2, self.sub_gate.num_qubits())

        _eig_comp = []
        theta_zero_exists = False
        for theta, projector in _sub_eig_comp:
            shifted_theta = theta + self.sub_gate_global_shift
            _expanded_projector = cirq.linalg.block_diag(
                np.diag([0] * (pow(2, self.num_qubits()) - sub_gate_shape)),
                projector
            )

            if shifted_theta == 0:
                _expanded_projector += np.diag(
                    [1] * (pow(2, self.num_qubits()) - sub_gate_shape)
                    + [0] * sub_gate_shape
                )
                theta_zero_exists = True

            _eig_comp.append(EigenComponent(shifted_theta, _expanded_projector))

        if theta_zero_exists is False:
            _eig_comp.append(EigenComponent(
                0,
                np.diag(
                    [1] * (pow(2, self.num_qubits()) - sub_gate_shape)
                    + [0] * sub_gate_shape
                )
            ))

        return _eig_comp

    def on(self, *qubits):
        _op = super(ControlledEigenGate, self).on(*qubits)
        return cirq.GateOperation(
            ControlledEigenGate(self.sub_gate,
                                num_controls=self.num_controls()),
            _op.qubits
        )

    def _value_equality_values_(self):
        return (
            self.num_controls(),
            self._canonical_exponent, self._global_shift
        )


class CRx(ControlledEigenGate):
    """
    A controlled Rx gate with the matrix_a Rx = e^{-i X rads / 2}.

    """

    def __init__(self, rads):
        self.rads = rads
        super().__init__(sub_gate=cirq.rx(rads),
                         num_controls=1)

    def _decompose_(self, qubits):
        control, target = qubits

        yield cirq.rz(np.pi / 2.0).on(target)
        yield cirq.CNOT(control=control, target=target)
        yield cirq.ry(-self.rads / 2.0).on(target)
        yield cirq.CNOT(control=control, target=target)
        yield cirq.ry(self.rads / 2.0).on(target)
        yield cirq.rz(-np.pi / 2.0).on(target)

    def on(self, control, target):
        return cirq.GateOperation(self, [control, target])

    def _is_parameterized_(self):
        return cirq.protocols.is_parameterized(self.rads)

    def parameters(self) -> typing.Optional[sympy.Symbol]:
        if self._is_parameterized_():
            return self.rads
        return None


class CRy(ControlledEigenGate):
    """
    A controlled Ry gate with the matrix_a Ry = e^{-i Y rads / 2}.

    """

    def __init__(self, rads):
        self.rads = rads
        super().__init__(sub_gate=cirq.ry(rads),
                         num_controls=1)

    def _decompose_(self, qubits):
        control, target = qubits

        yield cirq.CNOT(control=control, target=target)
        yield cirq.ry(-self.rads / 2.0).on(target)
        yield cirq.CNOT(control=control, target=target)
        yield cirq.ry(self.rads / 2.0).on(target)

    def on(self, control, target):
        return cirq.GateOperation(self, [control, target])

    def _is_parameterized_(self):
        return cirq.protocols.is_parameterized(self.rads)

    def parameters(self) -> typing.Optional[sympy.Symbol]:
        if self._is_parameterized_():
            return self.rads
        return None


class CRz(ControlledEigenGate):
    """
    A controlled Rz gate with the matrix_a Rz = e^{-i Z rads / 2}.

    """

    def __init__(self, rads):
        self.rads = rads
        super().__init__(sub_gate=cirq.rz(rads),
                         num_controls=1)

    def _decompose_(self, qubits):
        control, target = qubits

        yield cirq.rz(self.rads / 2.0).on(target)
        yield cirq.CNOT(control=control, target=target)
        yield cirq.rz(-self.rads / 2.0).on(target)
        yield cirq.CNOT(control=control, target=target)

    def on(self, control, target):
        return cirq.GateOperation(self, [control, target])

    def _is_parameterized_(self):
        return cirq.protocols.is_parameterized(self.rads)

    def parameters(self) -> typing.Optional[sympy.Symbol]:
        if self._is_parameterized_():
            return self.rads
        return None
