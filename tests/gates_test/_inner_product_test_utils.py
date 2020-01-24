import cirq
import numpy as np


def _test_inner_product(method):
    from paulicirq.gates.swap_test_gate import add_swap_test
    from paulicirq.gates.swap_test_gate import inner_product_from_swap_test_result

    from paulicirq.gates.overlap_test_gate import add_overlap_test
    from paulicirq.gates.overlap_test_gate import inner_product_from_overlap_test_result

    if method == "SWAP":
        add_test = add_swap_test
        inner_product_from_test_result = \
            inner_product_from_swap_test_result
    elif method == "overlap":
        add_test = add_overlap_test
        inner_product_from_test_result = \
            inner_product_from_overlap_test_result
    else:
        raise ValueError("Not supported test method.")

    circuit = cirq.Circuit()
    state1 = cirq.LineQubit.range(0, 3)
    state2 = cirq.LineQubit.range(3, 6)
    simulator = cirq.Simulator()

    for qubit1, qubit2 in zip(state1, state2):
        circuit.append([
            cirq.X(qubit1),

            cirq.X(qubit2),
            cirq.H(qubit2)
        ])

    measurement = add_test(
        state1,
        state2,
        circuit
    )

    print(circuit)

    result = simulator.run(circuit, repetitions=10000)
    # print(result)

    inner_product_simulator = inner_product_from_test_result(result, measurement)

    zero_state = np.array([1, 0])
    initial_state = cirq.kron(zero_state, zero_state, zero_state).flatten()

    _x = cirq.unitary(cirq.X)
    _h = cirq.unitary(cirq.H)

    all_x = cirq.kron(_x, _x, _x)
    all_h = cirq.kron(_h, _h, _h)

    _state1 = all_x @ initial_state
    _state2 = all_h @ all_x @ initial_state
    inner_product_exact = _state1 @ _state2

    return inner_product_simulator, inner_product_exact
