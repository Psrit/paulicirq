import re
import typing

import numpy as np
import cirq

from paulicirq.utils import get_all_measurement_keys


def add_overlap_test(
        state1: typing.Union[cirq.Qid, typing.Iterable[cirq.Qid]],
        state2: typing.Union[cirq.Qid, typing.Iterable[cirq.Qid]],
        circuit: cirq.Circuit
) -> str:
    """
    Add an overlap test between two quantum states `state1` and `state2`, and then add
    the test into `circuit`.

    0: ───────@───H───M('Overlap test measure 0 - pair 0')───
              │       │
    1: ───────X───────M──────────────────────────────────────

    :param state1:
        Quantum state to be tested.
    :param state2:
        Quantum state to be tested.
    :param circuit:
        The quantum circuit to which the overlap test is to be appended.
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

    measurement_name = "Overlap test measure"
    pattern = re.compile(
        r"^{} (\d+) - state ([12])$".format(measurement_name)
    )

    existed_overlap_measurement_ids = set()
    for key in get_all_measurement_keys(circuit):
        m = re.match(pattern, key)
        if m:
            existed_overlap_measurement_ids.add(int(m.group(1)))

    overlap_measurement_id = (max(existed_overlap_measurement_ids) + 1
                              if len(existed_overlap_measurement_ids) != 0
                              else 0)
    measurement_name += (" " + str(overlap_measurement_id))

    for i, (qubit1, qubit2) in enumerate(zip(state1, state2)):
        circuit.append([
            cirq.CNOT.on(qubit1, qubit2),
            cirq.H.on(qubit1)
        ])

        circuit.append([
            cirq.measure(qubit1, qubit2,
                         key=measurement_name + " - pair {}".format(i))
        ])

    return measurement_name


def inner_product_from_overlap_test_result(
        run_result: cirq.Result,
        overlap_measurement_key: str
) -> float:
    from collections import Counter
    overlap_result = 0.0
    for mkey, mresult in run_result.measurements.items():
        if mkey.startswith(overlap_measurement_key):
            postprocessed_outcome = np.logical_and(
                mresult[:, 0],  # qubit 1 in the pair
                mresult[:, 1]  # qubit 2 in the pair
            )
            overlap_result += postprocessed_outcome

    if isinstance(overlap_result, float):  # if overlap_result was not changed
        raise ValueError(
            "The overlap test key given does not match any measurement in the "
            "result, which is {}.".format(overlap_measurement_key)
        )

    overlap_result = typing.cast(np.ndarray, overlap_result)
    counter = Counter(overlap_result % 2)

    zeros = counter[0]  # the number of "pass"es in the overlap test
    prob_pass = float(zeros) / len(overlap_result)

    inner_product = (2.0 * prob_pass - 1.0) ** 0.5

    return inner_product
