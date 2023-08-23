import typing

import cirq

from paulicirq.op_tree import OpTreeGenerator

AnyNumberOfQids = typing.Any  # positional arguments like (q0, q1, q2) or (*qubits)


class GateBlock(cirq.Gate):
    def __init__(
            self,
            op_generator: OpTreeGenerator,
            **kwargs
    ):
        self._num_qubits = op_generator.num_qubits
        self._op_generator = op_generator
        self._generator_kwargs = kwargs

    def num_qubits(self) -> int:
        return self._num_qubits

    def _decompose_(self, qubits):
        yield self._op_generator(qubits, **self._generator_kwargs)

    def _is_parameterized_(self) -> bool:
        _circuit = cirq.Circuit()
        _qubits = cirq.LineQubit.range(self.num_qubits())
        _circuit.append(self._op_generator(_qubits, **self._generator_kwargs))
        return cirq.protocols.is_parameterized(_circuit)

    # Necessary for parametrized gate when optimized in an ansatz:
    def _resolve_parameters_(
        self, param_resolver: cirq.ParamResolver, recursive: bool = True
    ) -> "GateBlock":
        _resolved_op_generator = cirq.resolve_parameters(
            self._op_generator, param_resolver, recursive
        )

        return GateBlock(_resolved_op_generator, **self._generator_kwargs)

    def diagram(self) -> str:
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(self.num_qubits())
        circuit.append(self._op_generator(qubits, **self._generator_kwargs))

        return str(circuit)
