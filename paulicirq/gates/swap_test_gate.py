import typing

import cirq

from paulicirq.utils import get_all_measurement_keys, generate_auxiliary_qubit


def add_swap_test(
    state1: typing.Union[cirq.Qid, typing.Iterable[cirq.Qid]],
    state2: typing.Union[cirq.Qid, typing.Iterable[cirq.Qid]],
    circuit: cirq.Circuit,
    auxiliary_qubit: typing.Optional[cirq.Qid] = None,
    auxiliary_qubit_type: typing.Optional[
        typing.Union[typing.Type[cirq.GridQubit], typing.Type[cirq.LineQubit]]
    ] = cirq.GridQubit,
    cswap_in_elementary_gates: bool = False
) -> typing.Tuple[str, cirq.Qid]:
    """
    Add a SWAP test between two quantum states `state1` and `state2`, and then add
    the test into `circuit`.

    999: ───H───@───H───M('SWAP test measure')───
                │
    1: ─────────SWAP─────────────────────────────
                │
    2: ─────────SWAP─────────────────────────────

    :param state1:
        Quantum state to be tested.
    :param state2:
        Quantum state to be tested.
    :param circuit:
        The quantum circuit to which the SWAP test is to be appended.
    :param auxiliary_qubit:
        The auxiliary qubit.
        If being None, it will be created by `generate_auxiliary_qubit`.
        If being a `cirq.Qid` object, then it will be used as the auxiliary
        qubit of our SWAP test, and the argument `auxiliary_qubit_type` will be
        ignored.
        Note: `auxiliary_qubit` and `auxiliary_qubit_type` can not both be
        `None` at the same time.
    :param auxiliary_qubit_type:
        The type of the auxiliary qubit of SWAP test. Only LineQubit and
        GridQubit are supported.
        If `auxiliary_qubit` is None, this argument will be used to create the
        auxiliary qubit.
        If `auxiliary_qubit` is a `cirq.Qid` object, then this argument will be
        ignored.
        Note: `auxiliary_qubit` and `auxiliary_qubit_type` can not both be
        `None` at the same time.
    :param cswap_in_elementary_gates:
        Tells whether the CSWAP gate will be represented in elementary gates.
    :return:
        The string key of the measurement.

    """
    if isinstance(state1, cirq.Qid):
        state1 = [state1]
    if isinstance(state2, cirq.Qid):
        state2 = [state2]

    if len(state1) != len(state2):
        raise ValueError(
            "Qubit lengths of the two target states must equal, "
            "but {} != {}.".format(len(state1), len(state2))
        )

    if auxiliary_qubit is None:
        auxiliary_qubit = generate_auxiliary_qubit(circuit, auxiliary_qubit_type)
    measurement_name = "SWAP test measure "

    existed_swap_measurement_ids = {int(key[len(measurement_name):])
                                    for key in get_all_measurement_keys(circuit)
                                    if key.startswith(measurement_name)}
    swap_measurement_id = (max(existed_swap_measurement_ids) + 1
                           if len(existed_swap_measurement_ids) != 0
                           else 0)
    measurement_name += str(swap_measurement_id)

    circuit.append(cirq.H(auxiliary_qubit))
    for qubit1, qubit2 in zip(state1, state2):
        cswap_op = cirq.CSWAP.on(auxiliary_qubit, qubit1, qubit2)
        if cswap_in_elementary_gates:
            circuit.append(
                cirq.decompose(cswap_op)
            )
        else:
            circuit.append(cswap_op)

    circuit.append([
        cirq.H(auxiliary_qubit),
        cirq.measure(auxiliary_qubit, key=measurement_name)
    ])

    return measurement_name, auxiliary_qubit


def inner_product_from_swap_test_result(
    run_result: cirq.Result,
    swap_measurement_key: str
) -> float:
    from collections import Counter
    swap_result = (run_result.measurements[swap_measurement_key]).flatten()
    counter = Counter(swap_result)

    zeros = counter[0]  # the number of "pass"es in the SWAP test
    prob_pass = float(zeros) / len(swap_result)

    inner_product = (2.0 * prob_pass - 1.0) ** 0.5

    return inner_product
