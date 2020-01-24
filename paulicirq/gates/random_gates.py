from typing import Optional, Iterable, Tuple, Any, cast

import numpy as np
from cirq import linalg, protocols
from cirq._compat import proper_repr
from cirq.ops import raw_types, pauli_gates
from cirq.ops.matrix_gates import _matrix_to_diagram_symbol
from scipy.stats import unitary_group


# Copied from https://github.com/quantumlib/Cirq/blob/ce1d8297b92adf97780b375c9bcec1e1aaab103e/cirq/ops/matrix_gates.py#L26
# for interface compatibility with future versions.
class MatrixGate(raw_types.Gate):
    """A unitary qubit or qudit gate defined entirely by its matrix_a."""

    def __init__(self,
                 matrix: np.ndarray,
                 *,
                 qid_shape: Optional[Iterable[int]] = None) -> None:
        """Initializes a matrix_a gate.
        Args:
            matrix: The matrix_a that defines the gate.
            qid_shape: The shape of state tensor that the matrix_a applies to.
                If not specified, this value is inferred by assuming that the
                matrix_a is supposed to apply to qubits.
        """
        if qid_shape is None:
            if len(matrix.shape) != 2 or not matrix.shape[0]:
                raise ValueError('`matrix_a` must be a 2d numpy array.')
            n = int(np.round(np.log2(matrix.shape[0])))
            if 2 ** n != matrix.shape[0]:
                raise ValueError('Matrix width is not a power of 2 and '
                                 'qid_shape is not specified.')
            qid_shape = (2,) * n

        self._matrix = matrix
        self._qid_shape = tuple(qid_shape)
        m = int(np.prod(self._qid_shape))
        if self._matrix.shape != (m, m):
            raise ValueError('Wrong matrix_a shape for qid_shape.\n'
                             f'Matrix shape: {self._matrix.shape}\n'
                             f'qid_shape: {self._qid_shape}\n')

        if not linalg.is_unitary(matrix):
            raise ValueError(f'Not a unitary matrix_a: {self._matrix}')

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def __pow__(self, exponent: Any) -> 'MatrixGate':
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        e = cast(float, exponent)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b ** e)
        return MatrixGate(new_mat)

    def _phase_by_(self, phase_turns: float,
                   qubit_index: int) -> 'MatrixGate':
        if not isinstance(phase_turns, (int, float)):
            return NotImplemented
        if self._qid_shape[qubit_index] != 2:
            return NotImplemented
        gate = pauli_gates.Z ** (phase_turns * 2)
        result = np.copy(self._matrix).reshape(self._qid_shape * 2)

        p = np.exp(2j * np.pi * phase_turns)
        i = qubit_index
        j = qubit_index + len(self._qid_shape)
        result[linalg.slice_for_qubits_equal_to([i], 1)] *= p
        result[linalg.slice_for_qubits_equal_to([j], 1)] *= np.conj(p)
        return MatrixGate(matrix=result.reshape(self._matrix.shape),
                          qid_shape=self._qid_shape)

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        return np.copy(self._matrix)

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs'
                               ) -> 'protocols.CircuitDiagramInfo':
        main = _matrix_to_diagram_symbol(self._matrix, args)
        rest = [f'#{i + 1}' for i in range(1, len(self._qid_shape))]
        return protocols.CircuitDiagramInfo(wire_symbols=(main, *rest))

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((MatrixGate, vals))

    def _approx_eq_(self, other: Any, atol) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self._matrix, other._matrix, rtol=0, atol=atol)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.array_equal(self._matrix, other._matrix)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'cirq.MatrixGate({})'.format(proper_repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))


class RandomMatrixGate(MatrixGate):
    def __init__(self, num_qubits):
        self._num_qubits = num_qubits
        matrix = unitary_group.rvs(2 ** num_qubits)
        super(RandomMatrixGate, self).__init__(matrix)

    def num_qubits(self) -> int:
        return self._num_qubits

    def __repr__(self):
        return 'RandomMatrixGate({})'.format(proper_repr(self._matrix))
