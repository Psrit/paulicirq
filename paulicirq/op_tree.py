import collections
import copy
import typing
from abc import abstractmethod, ABCMeta

import cirq
import numpy as np
import sympy

from paulicirq.linear_combinations import LinearSymbolicDict


@cirq.value.value_equality(distinct_child_types=True)
class OpTreeGenerator(object):
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    @property
    @abstractmethod
    def num_qubits(self):
        pass

    @abstractmethod
    def __call__(
        self,
        qubits: typing.Sequence[cirq.Qid],
        **kwargs
    ) -> cirq.OP_TREE:
        pass

    def check_num_of_given_qubits(self, qubits):
        if self.num_qubits != len(qubits):
            raise ValueError(
                "The number of qubits ({}) != num_qubits of generator ({})"
                .format(len(qubits), self.num_qubits)
            )

    @abstractmethod
    def params(self) -> typing.Iterable[sympy.Symbol]:
        pass

    def _resolve_parameters_(
        self, param_resolver: cirq.ParamResolver, recursive: bool = True
    ):
        class _ParamResolvedGenerator(type(self)):
            def __call__(_self, qubits, **kwargs) -> cirq.OP_TREE:
                _op_tree = self.__call__(qubits, **kwargs)
                _resolved_op_tree = cirq.transform_op_tree(
                    _op_tree,
                    op_transformation=(
                        lambda op: cirq.resolve_parameters(op, param_resolver, recursive)
                    )
                )

                return _resolved_op_tree

            def params(self):
                return ()

        _resolved_generator = copy.deepcopy(self)
        _resolved_generator.__class__ = _ParamResolvedGenerator

        return _resolved_generator

    def _value_equality_values_(self):
        return id(self)

    def diagram(self, **call_kwargs) -> str:
        qubits = [
            cirq.NamedQubit("qubit {}".format(i)) for i in range(self.num_qubits)
        ]
        circuit = cirq.Circuit()
        circuit.append(self.__call__(qubits, **call_kwargs))

        return str(circuit)

    @staticmethod
    def join(
        generator1: "OpTreeGenerator",
        generator2: "OpTreeGenerator"
    ) -> "OpTreeGenerator":
        if generator1.num_qubits != generator2.num_qubits:
            raise ValueError(
                "`num_qubits` of the given two generators must equal, "
                "but {} != {}."
                .format(generator1.num_qubits, generator2.num_qubits)
            )

        kwargs1 = generator1._kwargs
        kwargs2 = generator2._kwargs
        for key in set.intersection(
            set(kwargs1.keys()), set(kwargs2.keys())
        ):
            if kwargs1[key] != kwargs2[key]:
                raise ValueError(
                    "Common keyword argument found in the given generators, "
                    "but the argument values don't equal: {} != {}."
                    .format(kwargs1[key], kwargs2[key])
                )

        kwargs = copy.deepcopy(kwargs1)
        kwargs.update(kwargs2)

        class _JoinedGenerator(type(generator1)):
            def __call__(
                self,
                qubits: typing.Sequence[cirq.Qid],
                **call_kwargs
            ) -> cirq.OP_TREE:
                yield generator1(qubits, **call_kwargs)
                yield generator2(qubits, **call_kwargs)

            def params(self) -> typing.Iterable[sympy.Symbol]:
                params1 = set(generator1.params())
                params2 = set(generator2.params())
                return params1.union(params2)

        return _JoinedGenerator(**kwargs1)


class VariableNQubitsGenerator(OpTreeGenerator, metaclass=ABCMeta):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits=num_qubits)
        self._num_qubits = num_qubits

    @property
    def num_qubits(self):
        return self._num_qubits


def simulate_generator(
    generator: OpTreeGenerator,
    generator_call_kwargs: dict,
    param_resolver: typing.Optional[cirq.ParamResolverOrSimilarType],
    qubit_order=cirq.ops.QubitOrder.DEFAULT,
    initial_state=None
) -> cirq.SimulationTrialResult:
    qubits = cirq.LineQubit.range(generator.num_qubits)
    circuit = cirq.Circuit()
    circuit.append(
        generator(
            qubits,
            **generator_call_kwargs
        )
    )
    simulator = cirq.Simulator(dtype=np.complex128)
    result = simulator.simulate(
        circuit, param_resolver,
        qubit_order, initial_state
    )

    return result


def lcg_grad(
    lcg: typing.OrderedDict[OpTreeGenerator, sympy.Basic],
    parameter: sympy.Symbol,
    generator_call_kwargs=None
) -> typing.OrderedDict[OpTreeGenerator, sympy.Basic]:
    from paulicirq.grad import op_tree_generator_grad
    if generator_call_kwargs is None:
        generator_call_kwargs = {}

    grad_dict = LinearSymbolicDict({})

    for _generator, _coeff in lcg.items():
        grad_dict += LinearSymbolicDict({
            _generator: sympy.diff(_coeff, parameter)
        })

        for _grad_generator, _grad_coeff in op_tree_generator_grad(
            _generator, parameter, **generator_call_kwargs
        ).items():
            grad_dict += LinearSymbolicDict({
                _grad_generator: _coeff * _grad_coeff
            })

    return collections.OrderedDict(grad_dict)
