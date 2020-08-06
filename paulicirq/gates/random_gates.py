from cirq._compat import proper_repr
from cirq.ops.matrix_gates import MatrixGate
from scipy.stats import unitary_group


class RandomMatrixGate(MatrixGate):
    def __init__(self, num_qubits):
        self._num_qubits = num_qubits
        matrix = unitary_group.rvs(2 ** num_qubits)
        super(RandomMatrixGate, self).__init__(matrix)

    def num_qubits(self) -> int:
        return self._num_qubits

    def __repr__(self):
        return 'RandomMatrixGate({})'.format(proper_repr(self._matrix))
