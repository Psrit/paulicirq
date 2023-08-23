import numbers
import typing

import cirq
import sympy


# TODO: add unit tests
class General1BitRotation(cirq.SingleQubitGate):
    def __init__(
        self,
        rad1, rad2, rad3,
        global_t: typing.Optional[
            typing.Union[sympy.Symbol, numbers.Number]
        ] = None
    ):
        """
        Create a general one-bit rotation gate, which is defined by

            U(rad1 * t, rad2 * t, rad3 * t) =
                Rz(rad3 * t) Ry(rad2 * t) Rz(rad1 * t),

        where t is the global exponent of rotation `global_t`.

        :param rad1:
            Rotation angle.
        :param rad2:
            Rotation angle.
        :param rad3:
            Rotation angle.
        :param global_t:
            The global exponent of rotation. Default to be 1.0.

        """
        t = global_t if global_t is not None else 1.0
        self.rad1 = rad1 * t
        self.rad2 = rad2 * t
        self.rad3 = rad3 * t

    def _is_parameterized_(self) -> bool:
        return (cirq.protocols.is_parameterized(self.rad1) or
                cirq.protocols.is_parameterized(self.rad2) or
                cirq.protocols.is_parameterized(self.rad3))

    def _decompose_(self, qubits: typing.Sequence[cirq.Qid]):
        q = qubits[0]
        yield cirq.rz(self.rad1).on(q)
        yield cirq.ry(self.rad2).on(q)
        yield cirq.rz(self.rad3).on(q)

    def _resolve_parameters_(
        self, param_resolver: cirq.ParamResolver, recursive: bool = True
    ) -> "General1BitRotation":
        return General1BitRotation(
            rad1=param_resolver.value_of(self.rad1, recursive),
            rad2=param_resolver.value_of(self.rad2, recursive),
            rad3=param_resolver.value_of(self.rad3, recursive)
        )

    def _circuit_diagram_info_(
        self, args: cirq.protocols.CircuitDiagramInfoArgs
    ) -> cirq.protocols.CircuitDiagramInfo:
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=("U({}, {}, {})"
                          .format(self.rad1, self.rad2, self.rad3),)
        )
