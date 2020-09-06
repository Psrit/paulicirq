from .controlled_gates import Controlled1BitMatrixGate
from .gate_block import GateBlock, AnyNumberOfQids
from .hadamard_test_gate import (
    add_hadamard_test,
    inner_product_from_hadamard_test_result
)
from .overlap_test_gate import (
    add_overlap_test,
    inner_product_from_overlap_test_result
)
from .pauli_exp_gates import (
    GlobalPhaseGate,
    TwoPauliExpGate,
    PauliWordExpGate
)
from .random_gates import RandomMatrixGate
from .swap_test_gate import (
    add_swap_test,
    inner_product_from_swap_test_result
)
