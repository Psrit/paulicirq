import copy
from abc import abstractmethod, ABCMeta

import cirq
import typing

import sympy


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
            qubits: typing.Iterable[cirq.Qid],
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

    def _resolve_parameters_(self, param_resolver: cirq.ParamResolver):
        class _ParamResolvedGenerator(type(self)):
            def __call__(_self, qubits, **kwargs) -> cirq.OP_TREE:
                _op_tree = self.__call__(qubits, **kwargs)
                _resolved_op_tree = cirq.transform_op_tree(
                    _op_tree,
                    op_transformation=(
                        lambda op: cirq.resolve_parameters(op, param_resolver)
                    )
                )

                return _resolved_op_tree

            def params(self):
                return ()

        _resolved_generator = copy.deepcopy(self)
        _resolved_generator.__class__ = _ParamResolvedGenerator

        return _resolved_generator

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
                    qubits: typing.Iterable[cirq.Qid],
                    **kwargs
            ) -> cirq.OP_TREE:
                yield generator1(qubits, **kwargs)
                yield generator2(qubits, **kwargs)

            def params(self) -> typing.Iterable[sympy.Symbol]:
                params1 = set(generator1.params())
                params2 = set(generator2.params())
                return params1.union(params2)

        kwargs = generator1._kwargs
        return _JoinedGenerator(**kwargs)


class VariableNQubitsGenerator(OpTreeGenerator, metaclass=ABCMeta):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits=num_qubits)
        self._num_qubits = num_qubits

    @property
    def num_qubits(self):
        return self._num_qubits
