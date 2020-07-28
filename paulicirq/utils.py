import math
import sys
import typing
import warnings

import cirq
import numpy as np
import sympy

TOutput = typing.Any


class lazy_load_instance_property(object):
    def __init__(self, property_func: typing.Callable[..., TOutput]):
        self.property_func = property_func

    def __get__(self, instance, owner) -> TOutput:
        value = self.property_func(instance)
        setattr(instance, self.property_func.__name__, value)
        return value


def get_all_measurement_keys(circuit: cirq.Circuit) -> set:
    all_measurement_keys = set()

    for moment in circuit:
        for op in moment:
            if cirq.is_measurement(op):
                all_measurement_keys.add(cirq.measurement_key(op))

    return all_measurement_keys


def get_all_line_qubit_ids(circuit: cirq.Circuit) -> tuple:
    """
    Return IDs of all LineQubits in `circuit` in order.

    """
    all_line_qubit_ids = set()

    for qubit in circuit.all_qubits():
        if isinstance(qubit, cirq.LineQubit):
            all_line_qubit_ids.add(qubit.x)

    return tuple(sorted(all_line_qubit_ids))


def generate_auxiliary_qubit(circuit: cirq.Circuit) -> cirq.LineQubit:
    """
    Generate an auxiliary qubit in `circuit`.

    :param circuit:
        The target circuit.
    :return:
        The generated auxiliary qubit, which is a LineQubit, and whose ID number
        is greater than the ones of all existing LineQubits in `circuit` and is
        also greater than 999.

    """
    existed_ids = get_all_line_qubit_ids(circuit)
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

    return cirq.LineQubit(_id)


def is_complex_close(
        a: complex, b: complex,
        rtol: typing.Optional[float] = 1e-09,
        atol: typing.Optional[float] = 0.0
) -> True:
    return (
            math.isclose(a.real, b.real, rel_tol=rtol, abs_tol=atol)
            and
            math.isclose(a.imag, b.imag, rel_tol=rtol, abs_tol=atol)
    )


class ToBeTested:
    def __init__(self, func, stream: typing.TextIO = sys.stderr):
        self._func = func
        self._stream = stream

    def __call__(self, *args, **kwargs):
        warnings.warn("Function {} needs to be tested.".format(self._func.__name__))
        return self._func(*args, **kwargs)


def pauli_expansion_for_any_matrix(matrix: np.ndarray) -> cirq.LinearDict[str]:
    class _HasUnitary:
        def __init__(self, matrix: np.ndarray):
            self._matrix = matrix

        def _unitary_(self) -> np.ndarray:
            return self._matrix

    return cirq.pauli_expansion(_HasUnitary(matrix))


def random_complex_matrix(*dn) -> np.ndarray:
    amp = np.random.rand(*dn)
    arg = np.random.rand(*dn) * 2 * np.pi
    matrix = amp * np.exp(1.0j * arg)
    return matrix


def inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculates the inner product between vectors a and b, which equals <a|b>.

    :param a:
        Vector of n-dimensional (n,), or some vectors of n-dimensional (m, n).
    :param b:
        Vector of n-dimensional (n,), or some vectors of n-dimensional (m, n).
    :return:
        Inner product between |a> and |b>.

    """
    braket_a_b = a.conjugate() * b
    if braket_a_b.ndim == 1:
        braket_a_b = sum(braket_a_b)
    else:
        braket_a_b = np.sum(braket_a_b, axis=1)
    return braket_a_b


def normalized_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculates the overlap between the vectors normalized from a and b.

    The overlap is also called fidelity, which equals |<a'|b'>|^2, where
    |a'> = a / ||a||, |b'> = b / ||b||.

    :param a:
        Vector of n-dimensional (n,), or some vectors of n-dimensional (m, n).
    :param b:
        Vector of n-dimensional (n,), or some vectors of n-dimensional (m, n).
    :return:
        Overlap between |a'> and |b'>.

    """
    braket_a_b = inner_product(a, b)  # (m,) or ()
    braket_a_b_squared = braket_a_b * braket_a_b.conjugate()  # |<a|b>|^2

    a_2 = inner_product(a, a)  # ||a||^2, (m,) or ()
    b_2 = inner_product(b, b)  # ||b||^2, (m,) or ()
    overlap = braket_a_b_squared / (a_2 * b_2)  # |<a'|b'>|^2
    return overlap


def resolve_scalar(c, param_resolver: cirq.ParamResolverOrSimilarType):
    try:
        _c_resolved = cirq.resolve_parameters(c, param_resolver)
    except TypeError:
        assert isinstance(c, sympy.Basic)
        c = c.subs(cirq.ParamResolver(param_resolver).param_dict)
        _c_resolved = complex(c)

    return _c_resolved


def deduplicate(sequence: typing.Sequence):
    """
    Remove repeated terms in `sequence` with the original order preserved.

    :param sequence:
        The sequence to be processed.
    :return:
        The processed sequence.

    """
    sequence_type = type(sequence)

    _set = set()
    _list = list(sequence)
    _deduplicated = []

    for term in _list:
        if term not in _set:
            _deduplicated.append(term)
            _set.add(term)

    _deduplicated = sequence_type(_deduplicated)
    return _deduplicated
