import sys
import typing
import warnings

TOutput = typing.Any


class lazy_load_instance_property(object):
    def __init__(self, property_func: typing.Callable[..., TOutput]):
        self.property_func = property_func

    def __get__(self, instance, owner) -> TOutput:
        value = self.property_func(instance)
        setattr(instance, self.property_func.__name__, value)
        return value


class ToBeTested:
    def __init__(self, func, stream: typing.TextIO = sys.stderr):
        self._func = func
        self._stream = stream

    def __call__(self, *args, **kwargs):
        warnings.warn("Function {} needs to be tested.".format(self._func.__name__))
        return self._func(*args, **kwargs)


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
