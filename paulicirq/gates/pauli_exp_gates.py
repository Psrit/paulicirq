import numbers
import typing
from typing import List, Union, Tuple

import cirq
import numpy as np
import scipy.linalg as splinalg
import sympy
from cirq.ops.eigen_gate import EigenComponent

from paulicirq.algorithms.pauliword import pauli_word_exp_factorization
from paulicirq.pauli import PauliWord, Pauli


def Rzz(rads: typing.Union[float, sympy.Symbol]):
    """Returns a gate with the matrix_a exp(-i Z⊗Z rads)."""
    pi = sympy.pi if isinstance(rads, sympy.Basic) else np.pi
    return cirq.ZZPowGate(exponent=2 * rads / pi, global_shift=-0.5)


class GlobalPhaseGate(cirq.SingleQubitGate, cirq.EigenGate):
    """
    Returns a gate which adds a global phase shift exp(i pi rad)
    to a qubit.

    """

    def __init__(self, rad: typing.Union[float, sympy.Basic]):
        self.rad = rad
        super(GlobalPhaseGate, self).__init__(exponent=rad)

    def _eigen_components(self) -> List[Union[EigenComponent,
                                              Tuple[float, np.ndarray]]]:
        return [(1, np.identity(2))]

    def _circuit_diagram_info_(
            self, args: cirq.protocols.CircuitDiagramInfoArgs
    ) -> cirq.protocols.CircuitDiagramInfo:
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=("e^{{({})πi}}".format(self.rad),)
        )

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _unitary_(self) -> np.ndarray:
        if self._is_parameterized_():
            return NotImplemented
        r = typing.cast(float, self.rad)
        return np.exp(1.0j * np.pi * r) * np.identity(2)

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(self.rad)

    def _decompose_(self, qubits: typing.Sequence[cirq.Qid]):
        yield cirq.ZPowGate(exponent=self.rad).on(qubits[0])
        yield cirq.X.on(qubits[0])
        yield cirq.ZPowGate(exponent=self.rad).on(qubits[0])
        yield cirq.X.on(qubits[0])

    # Necessary for parametrized gate when optimized in an ansatz:
    def _resolve_parameters_(
        self, param_resolver: cirq.ParamResolver, recursive: bool = True
    ) -> "GlobalPhaseGate":
        return GlobalPhaseGate(
            rad=param_resolver.value_of(self.rad, recursive)
        )


class TwoPauliExpGate(cirq.TwoQubitGate):
    """
    Two-qubit gate with the matrix_a form being
    $$
        e^{-i A0 A1 rad},
    $$
    where A0 and A1 denote the Pauli matrices {X, Y, Z}.

    """

    def __init__(
            self,
            pauli0: typing.Union[Pauli, str, numbers.Integral, np.ndarray],
            pauli1: typing.Union[Pauli, str, numbers.Integral, np.ndarray],
            rad: typing.Union[float, sympy.Basic]
    ):
        """
        Initialize the gate.

        :param pauli0:
            The `A0` in e^{-i A0 A1 rad}.
        :param pauli1:
            The `A1` in e^{-i A0 A1 rad}.
        :param rad:
            The `rad` in e^{-i A0 A1 rad}.

        """
        self.pauli0 = pauli0 if isinstance(pauli0, Pauli) else Pauli(pauli0)
        self.pauli1 = pauli1 if isinstance(pauli1, Pauli) else Pauli(pauli1)
        self.rad = rad

    def num_qubits(self) -> int:
        return 2

    def _circuit_diagram_info_(
            self, args: cirq.protocols.CircuitDiagramInfoArgs
    ) -> cirq.protocols.CircuitDiagramInfo:
        rad_str = ("{:.3}".format(self.rad) if isinstance(self.rad, float)
                   else self.rad)
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=(
                "^{}".format(self.pauli0),
                "^{0}{{{1}}}".format(self.pauli1, rad_str)
            )
        )

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _unitary_(self) -> np.ndarray:
        if self._is_parameterized_():
            return NotImplemented
        r = typing.cast(float, self.rad)
        return splinalg.expm(-1j * r *
                             np.kron(self.pauli0.array, self.pauli1.array))

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(self.rad)

    def _decompose_(self, qubits: typing.Sequence[cirq.Qid]):
        left_gates0, right_gates0 = self._transformation_gates_from_z(self.pauli0)
        left_gates1, right_gates1 = self._transformation_gates_from_z(self.pauli1)

        for r_gate0 in reversed(right_gates0):
            yield r_gate0.on(qubits[0])

        for r_gate1 in reversed(right_gates1):
            yield r_gate1.on(qubits[1])

        yield Rzz(self.rad).on(*qubits)

        for l_gate0 in reversed(left_gates0):
            yield l_gate0.on(qubits[0])

        for l_gate1 in reversed(left_gates1):
            yield l_gate1.on(qubits[1])

    @staticmethod
    def _transformation_gates_from_z(pauli: Pauli) -> typing.Tuple[
        typing.List[cirq.Gate], typing.List[cirq.Gate]
    ]:
        if pauli == Pauli("Z"):
            return [], []

        elif pauli == Pauli("X"):
            return [cirq.H], [cirq.H]

        elif pauli == Pauli("Y"):
            # vectors_y.inv.dagger @ z @ vectors_y.inv == y
            # vectors_y == i Rz(pi/2) Ry(pi/2) Rz(3pi/2)
            # vectors_y.inv == -i Rz(-3pi/2) Ry(-pi/2) Rz(-pi/2)
            # vectors_y.inv.dagger == i Rz(pi/2) Ry(pi/2) Rz(3pi/2)
            return (
                [cirq.rz(np.pi / 2), cirq.ry(np.pi / 2), cirq.rz(3 * np.pi / 2)],
                [cirq.rz(-3 * np.pi / 2), cirq.ry(-np.pi / 2), cirq.rz(-np.pi / 2)]
            )

        else:
            raise ValueError("invalid Pauli {}".format(pauli))

    # Necessary for parametrized gate when optimized in an ansatz:
    def _resolve_parameters_(
        self, param_resolver: cirq.ParamResolver, recursive: bool = True
    ) -> "PauliWordExpGate":
        return TwoPauliExpGate(
            pauli0=self.pauli0,
            pauli1=self.pauli1,
            rad=param_resolver.value_of(self.rad, recursive),
        )


class PauliWordExpGate(cirq.Gate):
    """
    Multi-qubit gate with the matrix_a from being
    $$
        e^{−i t P},
    $$
    where P denotes a Pauli word, i.e. direct product of Pauli matrices which
    act on different qubits.

    """

    def __init__(
            self,
            coefficient: typing.Union[float, sympy.Basic],
            pauli_word: PauliWord
    ):
        """
        Initialize the gate.

        :param coefficient:
            The `t` in e^{−i t P}.
        :param pauli_word:
            The `P` in e^{−i t P}.

        """
        self.coefficient = coefficient
        self.pauli_word = pauli_word

    def num_qubits(self) -> int:
        """The number of qubits this gate acts on."""
        # Note that the lengths of the Pauli words in the factorized exp gates
        # may vary, but they are always <= len(self.pauli_word) (which means
        # the number of active qubits in the circuit).
        return len(self.pauli_word)

    def _circuit_diagram_info_(
            self, args: cirq.protocols.CircuitDiagramInfoArgs
    ) -> cirq.protocols.CircuitDiagramInfo:
        wire_symbols = ["^{}".format(w) for w in self.pauli_word.fullstr]
        wire_symbols[-1] += (
            "{{{:.3}}}".format(self.coefficient) if isinstance(self.coefficient, float)
            else "{{{}}}".format(self.coefficient)
        )
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=tuple(wire_symbols)
        )

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(self.coefficient)

    def _decompose_(self, qubits: typing.Sequence[cirq.Qid]):
        factor = pauli_word_exp_factorization(self.coefficient, self.pauli_word)

        for coeff, word in reversed(factor):
            if word.effective_len == 1:
                # Note that `dict_form` is a property, therefore `word` won't be changed:
                qubit_index, rot_axis = word.dict_form.popitem()
                rotation_gate = getattr(
                    cirq, "r{}".format(rot_axis.lower())
                )  # type: typing.Union[cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate]
                # `* 2` comes from the 2 in the denominator in the exponential of R{x, y, z}
                yield rotation_gate(coeff * 2).on(qubits[qubit_index])

            else:  # word.effective_len == 2:
                (qubit_index0, rot_axis0), (qubit_index1, rot_axis1) = \
                    list(word.dict_form.items())

                two_qubit_gate = TwoPauliExpGate(Pauli(rot_axis0), Pauli(rot_axis1), coeff)
                yield two_qubit_gate.on(
                    qubits[qubit_index0], qubits[qubit_index1]
                )

    # Necessary for parametrized gate when optimized in an ansatz:
    def _resolve_parameters_(
        self, param_resolver: cirq.ParamResolver, recursive: bool = True
    ) -> "PauliWordExpGate":
        return PauliWordExpGate(
            coefficient=param_resolver.value_of(self.coefficient, recursive),
            pauli_word=self.pauli_word
        )
