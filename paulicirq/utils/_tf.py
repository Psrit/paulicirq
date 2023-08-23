import typing

import tensorflow as tf


def tf_isclose(
    a: tf.Tensor, b: tf.Tensor,
    rtol: typing.Optional[float] = None,
    atol: typing.Optional[float] = None
) -> tf.Tensor:
    """
    For all element pairs of (a, b), determine whether the two floating point
    numbers are close according to:

        abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    :param a:
        Input tensor `a`.
    :param b:
        Input tensor `b`.
    :param rtol:
        Maximum difference for being considered "close", relative to the
        magnitude of the input values (i.e. relative tolerance).
        Default to be 1e-09.
    :param atol:
        Maximum difference for being considered "close", regardless of the
        magnitude of the input values (i.e. absolute tolerance).
        Default to be 0.0.
    :return:
        The tensor containing all compare results.

    """
    import math

    if rtol is None:
        rtol = 1e-09
    if atol is None:
        atol = 0.0

    if len(a.shape) != len(b.shape):
        raise ValueError("The ndims of input tensors do not match: "
                         f"len({a.shape}) != len({b.shape})")

    if len(a.shape) > 1:
        def fn(x):
            return tf_isclose(x[0], x[1], rtol=rtol, atol=atol)
    else:
        def fn(x):
            return math.isclose(x[0], x[1], rel_tol=rtol, abs_tol=atol)
    return tf.map_fn(fn, elems=(a, b), dtype=tf.bool)


def tf_allclose(
    a: tf.Tensor, b: tf.Tensor,
    rtol: typing.Optional[float] = 1e-09,
    atol: typing.Optional[float] = 0.0
) -> bool:
    isclose = tf_isclose(a, b, rtol, atol)
    is_allclose = tf.reduce_all(isclose)
    return is_allclose.numpy()


def substates(
    states: tf.Tensor,
    keep_indices: typing.List[int],
    atol: typing.Union[int, float] = 1e-8
) -> tf.Tensor:
    def _substate(
        one_state: tf.Tensor
    ) -> tf.Tensor:
        import cirq
        one_substate = cirq.sub_state_vector(
            one_state.numpy(), keep_indices=keep_indices, atol=atol
        )
        return tf.convert_to_tensor(one_substate)

    substates_tensor = tf.map_fn(
        _substate, elems=states
    )
    return substates_tensor
