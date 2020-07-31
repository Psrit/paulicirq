import cirq
from cirq.protocols import decompose_once

from paulicirq.gates import GlobalPhaseGate

UNIVERSAL_GATE_SET = {
    lambda g: g == cirq.CNOT,
    lambda g: isinstance(g, cirq.XPowGate) and g._global_shift == -0.5,
    lambda g: isinstance(g, cirq.YPowGate) and g._global_shift == -0.5,
    lambda g: isinstance(g, cirq.ZPowGate) and g._global_shift == -0.5,
    lambda g: isinstance(g, GlobalPhaseGate)
}


def is_a_basic_gate(gate: cirq.Gate) -> bool:
    for check in UNIVERSAL_GATE_SET:
        if check(gate) is True:
            return True

    return False


def is_a_basic_operation(operation: cirq.GateOperation) -> bool:
    for check in UNIVERSAL_GATE_SET:
        if check(operation.gate) is True:
            return True

    return False


def is_an_indecomposable_operation(operation: cirq.GateOperation) -> bool:
    if decompose_once(operation, default=NotImplemented) is not NotImplemented:
        return False

    # TODO: add new decomposers
    return True
