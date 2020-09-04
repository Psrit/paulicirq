import functools
import signal
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


def deprecated(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        warnings.warn(
            "Function {} is deprecated.".format(func.__name__),
            DeprecationWarning
        )
        return func(*args, **kwargs)

    return wrapped


def complex_format(c: complex, template: str = "{:.2f}"):
    s = (template.format(c.real) +
         ("-" if c.imag < 0 else "+") +
         template.format(abs(c.imag)) + "i")
    return s


def set_timeout(timeout, *, callback, callback_kwargs=None):
    if callback_kwargs is None:
        callback_kwargs = {}

    def wrap_func(func):
        def handle(signum, frame):
            raise TimeoutError(
                f"Function {func} did not finish running in {timeout} seconds."
            )

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(timeout)  # start alarm signal
                result = func(*args, **kwargs)
                signal.alarm(0)  # close alarm signal
                return result
            except TimeoutError as e:
                if callback:
                    return callback(**callback_kwargs)

        return wrapped

    return wrap_func
