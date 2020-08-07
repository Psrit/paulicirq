import math
import typing

import numpy as np


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
