"""
Util
----

Collection of utility functions

"""

from math import factorial
from math import prod

from functools import partial

from typing import Iterable
from typing import Iterator
from typing import Callable

def multinomial(*sequence:tuple[int, ...]) -> float:
    """
    Compute multinomial coefficient for a given sequence (n, m, ...) of non-negative integers
    (n + m + ...)! / (n! * m! * ... )

    Parameters
    ----------
    *sequence: tuple[int, ...], non-negative
        input sequence of integers

    Returns
    ------
    float

    Examples
    --------
    >>> multinomial(2, 0)
    1.0
    >>> multinomial(1, 1)
    2.0
    >>> multinomial(0, 2)
    1.0

    """
    return factorial(sum(sequence)) / prod(map(factorial, sequence))


def flatten(array:Iterable, *, target:type=tuple) -> Iterator:
    """
    Flatten a nested tuple (or other selected target type container)

    Parameters
    ----------
    array: Iterable
        input nested iterable
    target: type, default=tuple
        target iterable type to flatten

    Yields
    ------
    Iterator

    Examples
    --------
    >>> [*flatten((1, (1, (1, (1, 1), 1)), ((1), (1))), target=tuple)]
    [1, 1, 1, 1, 1, 1, 1, 1]
    >>> [*flatten([1, [1, [1, [1, 1], 1]], [[1], [1]]], target=list)]
    [1, 1, 1, 1, 1, 1, 1, 1]

    """
    if isinstance(array, target):
        for element in array:
            yield from flatten(element, target=target)
    else:
        yield array


def curry_apply(function:Callable, table:tuple[int, ...], *pars:tuple) -> Callable:
    """
    Curry apply

    Given f(x, y, ...) and table = map(len, (x, y, ...)) return g(*x, *y, ...) = f(x, y, ...)

    Parameters
    ----------
    function: Callable
        input function
    table: tuple[int, ...]
        map(len, (x, y, ...))
    *pars: tuple
        passed to input function

    Returns
    ------
    Callable

    Examples
    --------
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2, y3 = y
    ...    return x1*x2*y1*y2*y3
    >>> def gn(x1, x2, y1, y2, y3):
    ...    return fn((x1, x2), (y1, y2, y3))
    >>> x, y = (1, 1), (1, 1, 1)
    >>> gn(*x, *y) == curry_apply(fn, (2, 3))(*x, *y)
    True

    """
    def clouser(*args:tuple):
        start = 0
        vecs = []
        for length in table:
            vecs.append(args[start:start + length])
            start += length
        return function(*vecs, *pars)
    return partial(clouser)
