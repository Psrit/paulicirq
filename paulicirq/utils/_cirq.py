import math
import typing

import cirq
import numpy as np
import sympy


def get_all_measurement_keys(circuit: cirq.Circuit) -> set:
    all_measurement_keys = set()

    for moment in circuit:
        for op in moment:
            if cirq.is_measurement(op):
                all_measurement_keys.add(cirq.measurement_key(op))

    return all_measurement_keys


def get_all_line_qubit_ids(circuit: cirq.Circuit) -> typing.Tuple[int]:
    """
    Return IDs of all LineQubits in `circuit` in order.

    """
    all_line_qubit_ids = set()

    for qubit in circuit.all_qubits():
        if isinstance(qubit, cirq.LineQubit):
            all_line_qubit_ids.add(qubit.x)

    return tuple(sorted(all_line_qubit_ids))


def get_all_grid_qubit_ids(circuit: cirq.Circuit) -> typing.Tuple[
    typing.Tuple[int], typing.Tuple[int]
]:
    """
    Return IDs (i.e. row numbers and column numbers) of all GridQubits in
    `circuit` in order.

    """
    all_grid_qubit_rows = set()
    all_grid_qubit_cols = set()

    for qubit in circuit.all_qubits():
        if isinstance(qubit, cirq.GridQubit):
            all_grid_qubit_rows.add(qubit.row)
            all_grid_qubit_cols.add(qubit.col)

    return (
        tuple(sorted(all_grid_qubit_rows)), tuple(sorted(all_grid_qubit_cols))
    )


def generate_auxiliary_qubit(
    circuit: cirq.Circuit,
    auxiliary_qubit_type: typing.Union[
        typing.Type[cirq.LineQubit], typing.Type[cirq.GridQubit]
    ] = cirq.LineQubit
):
    """
    Generate an auxiliary qubit in `circuit`.

    :param circuit:
        The target circuit.
    :param auxiliary_qubit_type:
        Designates the type of the generated auxiliary qubit. Only `LineQubit`
        and `GridQubit` are supported.
    :return:
        The generated auxiliary qubit, which is a LineQubit, and whose ID number
        is greater than the ones of all existing LineQubits in `circuit` and is
        also greater than 999.

    """

    def _generate_new_id(existed_ids):
        """
        Generate a new ID which does not exist in `existed_ids`. An ID can be
        the ID of a LineQubit, or the row/column number of a GridQubit.

        :param existed_ids:
            The tuple of existed IDs.
        :return:
            A new ID.

        """
        if len(existed_ids) == 0:  # if there is no qubit existed in the circuit:
            max_id = 1
        else:
            max_id = max(existed_ids)
            if max_id <= 0:  # to avoid math domain error for log10
                max_id = 1

        largest_digits = math.floor(math.log10(max_id)) + 1  # >= 1

        _id = 10 ** (largest_digits * 2) - 1
        if _id < 999:
            _id = 999

        return _id

    if auxiliary_qubit_type == cirq.LineQubit:
        existed_ids = get_all_line_qubit_ids(circuit)
        _id = _generate_new_id(existed_ids)
        q_aux = cirq.LineQubit(_id)

    elif auxiliary_qubit_type == cirq.GridQubit:
        existed_rows, existed_cols = get_all_grid_qubit_ids(circuit)
        _row = _generate_new_id(existed_rows)
        _col = _generate_new_id(existed_cols)
        q_aux = cirq.GridQubit(_row, _col)

    else:
        raise TypeError(f"Unsupported qubit type: {auxiliary_qubit_type}. "
                        f"Only LineQubit and GridQubit are supported.")

    return q_aux


def pauli_expansion_for_any_matrix(matrix: np.ndarray) -> cirq.LinearDict[str]:
    class _HasUnitary:
        def __init__(self, matrix: np.ndarray):
            self._matrix = matrix

        def _unitary_(self) -> np.ndarray:
            return self._matrix

    return cirq.pauli_expansion(_HasUnitary(matrix))


def resolve_scalar(
    c, param_resolver: cirq.ParamResolverOrSimilarType, recursive: bool = True
):
    try:
        _c_resolved = cirq.resolve_parameters(c, param_resolver, recursive)
    except TypeError:
        assert isinstance(c, sympy.Basic)
        c = c.subs(cirq.ParamResolver(param_resolver).param_dict)
        _c_resolved = complex(c)

    return _c_resolved


GateBatch = typing.List[typing.List[cirq.Gate]]
GateOperationBatch = typing.List[cirq.OP_TREE]
CircuitBatch = typing.List[cirq.Circuit]


def generate_random_rotation_batch(
    num_qubits, batch_size
) -> typing.List[typing.List['General1BitRotation']]:
    """
    Generate a batch of random rotations which act on `num_qubits` qubits.
    The batch size is designated by `batch_size`.

    :param num_qubits:
        The number of qubits the rotations will act on.
    :param batch_size:
        The batch size.
    :return:
        A batch of random rotations, i.e. a batch of 1-qubit rotations whose
        shape is (batch_size, num_qubits). Any term in the batch stands for a
        series of random rotations --- each of them can be used to act on a
        qubit.

    """
    from paulicirq.gates.general_rotation import General1BitRotation

    rotation_batch = []
    for i in range(batch_size):
        rotations = [
            General1BitRotation(
                *(np.random.uniform(size=[3, ], low=-np.pi, high=np.pi)),
                global_t=1.0
            )
            for _ in range(num_qubits)
        ]
        rotation_batch.append(rotations)

    return rotation_batch


def act_gate_batch_on_qubits(
    gate_batch: GateBatch,
    qubits: typing.Iterable[cirq.Qid],
    decompose_in_elementary_ops: bool = True,
    as_circuits: bool = True
) -> typing.Union[GateOperationBatch, CircuitBatch]:
    """
    Act a batch of gates on `qubits`.

    :param gate_batch:
        A batch of gates, whose shape is (batch_size, num_qubits). Any term
        in the batch contains `num_qubits` 1-qubit gates, which will be acted
        on each qubit in `qubits` in order.
    :param qubits:
        The qubits which the gates will act on.
    :param decompose_in_elementary_ops:
        Tells whether the gates in `gate_batch` should be decomposed into
        elementary operations.
    :param as_circuits:
        Tells whether the applied operations will be constructed as circuits.
        If False, the function will return a batch of series of operations;
        if True,  the function will return a batch of circuits.
    :return:
        The result of `gate_batch` being acted on `qubits`. See `as_circuits`
        for more details.

    """
    op_batch = []
    for gate_series in gate_batch:
        op = []
        for gate, q in zip(gate_series, qubits):
            if decompose_in_elementary_ops:
                op.append(cirq.decompose(gate(q)))
            else:
                op.append(gate(q))

        if as_circuits:
            op_batch.append(cirq.Circuit(op))
        else:
            op_batch.append(op)

    return op_batch


def replace_qubits(
    circuit: cirq.Circuit,
    qubit_map: typing.Mapping
):
    old_qubits = set(qubit_map.keys())
    new_qubits = set(qubit_map.values())
    if len(old_qubits) != len(new_qubits):
        raise ValueError(
            "`qubit_map` must be a bijective mapping."
        )

    qubits_in_circuit = circuit.all_qubits()
    unknown_old_qubits = old_qubits - qubits_in_circuit
    if unknown_old_qubits:
        raise ValueError(
            "Some qubits in `old_qubits` do not exist in the original circuit: "
            f"{unknown_old_qubits}"
        )

    new_circuit = cirq.Circuit()
    for moment in circuit:
        new_moment = cirq.Moment()
        for operation in moment:
            new_operation = operation.gate.on(
                *(qubit_map[q] for q in operation.qubits)
            )
            new_moment += new_operation
        new_circuit += new_moment

    return new_circuit
