import typing

import cirq

from paulicirq.utils import generate_auxiliary_qubit, get_all_measurement_keys


def add_hadamard_test(
        state: typing.Union[cirq.Qid, typing.Iterable[cirq.Qid]],
        gate_v: cirq.Gate,
        gate_a: cirq.Gate,
        is_imaginary_part: bool,
        circuit: cirq.Circuit
) -> str:
    """
    Add an Hadamard test for `state` |ψ0> to estimate <ψ0|V+ A V|ψ0>,
    the circuit of which is:

    999: ───H─────S/I─────@──────H──────M('Hadamard test measure 0')───
                          │
    0: ─────V─────────────A────────────────────────────────────────────

    where S/I is set to be the identity gate / the Clifford S gate when
    `is_imaginary_part` is False / True (i.e. when estimating the real /
    imaginary part of <ψ0|V+ A V|ψ0>).

    :param state:
        The input state |ψ0>.
    :param gate_v:
    :param gate_a:
    :param is_imaginary_part:
    :param circuit:
    :return:
    """
    if isinstance(state, cirq.Qid):
        state = [state]

    if gate_a.num_qubits() == gate_v.num_qubits() == len(state):
        pass
    else:
        raise ValueError(
            "The number of qubits acted on by gate `V` and `A` must equal to "
            "that of `state`, but they are found to be {}, {} and {}.".format(
                gate_v.num_qubits(), gate_a.num_qubits(), len(state)
            )
        )

    auxiliary_qubit = generate_auxiliary_qubit(circuit)

    measurement_name = "Hadamard test measure "

    existed_hadamard_measurement_ids = {int(key[len(measurement_name):])
                                        for key in get_all_measurement_keys(circuit)
                                        if key.startswith(measurement_name)}
    hadamard_measurement_id = (max(existed_hadamard_measurement_ids) + 1
                               if len(existed_hadamard_measurement_ids) != 0
                               else 0)
    measurement_name += str(hadamard_measurement_id)

    circuit.append([
        cirq.H(auxiliary_qubit),
        gate_v.on(*state)
    ])

    if is_imaginary_part is False:
        pass
    else:
        circuit.append(cirq.S(auxiliary_qubit))

    circuit.append([
        cirq.ControlledGate(gate_a).on(auxiliary_qubit, *state)
    ])

    circuit.append(cirq.H(auxiliary_qubit))
    circuit.append(cirq.measure(auxiliary_qubit, key=measurement_name))

    return measurement_name


def inner_product_from_hadamard_test_result(
        run_result: cirq.Result,
        hadamard_measurement_key: str,
        is_imaginary_part: bool
) -> float:
    from collections import Counter
    hadamard_result = (run_result.measurements[hadamard_measurement_key]).flatten()
    counter = Counter(hadamard_result)

    zeros = counter[0]
    prob_zero = float(zeros) / len(hadamard_result)

    if is_imaginary_part is False:
        inner_product_re = 2.0 * prob_zero - 1.0
        return inner_product_re
    else:
        inner_product_im = 1.0 - 2.0 * prob_zero
        return inner_product_im
