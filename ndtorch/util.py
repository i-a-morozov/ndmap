"""
Util
----

Collection of utility functions

"""

from math import factorial
from math import prod

from functools import partial

from typing import TypeAlias
from typing import Union
from typing import Iterable
from typing import Iterator
from typing import Callable
from typing import Any

from multimethod import multimethod

import numpy
from numpy import ndarray as Array

import torch
from torch import Tensor


State       : TypeAlias = Tensor
Knobs       : TypeAlias = list[Tensor]
Point       : TypeAlias = list[Tensor]
Delta       : TypeAlias = list[Tensor]
Table       : TypeAlias = list
Series      : TypeAlias = dict[tuple[int, ...], Tensor]
Signature   : TypeAlias = Union[list[tuple[int, ...]], list[tuple[tuple[int, ...], float]]]
Mapping     : TypeAlias = Callable
Observable  : TypeAlias = Callable
Hamiltonian : TypeAlias = Callable


def multinomial(*sequence:int) -> float:
    """
    Compute multinomial coefficient for a given sequence (n, m, ...) of non-negative integers
    (n + m + ...)! / (n! * m! * ... )

    Parameters
    ----------
    *sequence: int, non-negative
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


def nest(power:int, function:Callable, *pars:tuple) -> Callable:
    """
    Generate nested function

    Parameters
    ----------
    power : int
        nest power
    function : Callable
        function to nest
    *pars: tuple
        fixed parameters

    Returns
    -------
    Callable

    Examples
    --------
    >>> nest(5, lambda x: x**2)(2)
    4294967296

    """
    def wrapper(x, *pars):
        for _ in range(power):
            x = function(x, *pars)
        return x
    return wrapper


def orthogonal(n:int,
               m:int, *,
               dtype:torch.dtype=torch.float64,
               device:torch.device=torch.device('cpu'),
               **kwargs) -> Tensor:
    """
    Generate random orthonormal (n x m) matrix
    
    Parameters
    ----------
    n, m: int
        n, m
    dtype: torch.dtype, default=torch.float64
        output type
    device: torch.device, torch.device=torch.device('cpu')
        output device
    **kwargs: dict
        passed to torch.linalg.svd function

    Returns
    -------
    Tensor

    Examples
    --------
    >>> import torch
    >>> torch.manual_seed(1)
    >>> orthogonal(4, 4)
    tensor([[-0.4048, -0.7515, -0.5066, -0.1216],
            [ 0.1141, -0.5599,  0.6068,  0.5525],
            [ 0.1797,  0.1702, -0.5821,  0.7746],
            [-0.8893,  0.3046,  0.1909,  0.2827]], dtype=torch.float64)

    """
    u, _, vh = torch.linalg.svd(torch.randn((n, m), dtype=dtype, device=device, **kwargs))
    return u @ vh


def symplectic(state:Tensor) -> Tensor:
    """
    Generate symplectic identicy matrix for a given state

    Parameters
    ----------
    state: Tensor
        state

    Returns
    -------
    Tensor

    Examples
    --------
    >>> import torch
    >>> state = torch.tensor([0.0, 0.0])
    >>> symplectic(state)
    tensor([[ 0.,  1.],
            [-1.,  0.]])
    >>> state = torch.tensor([0.0, 0.0, 0.0, 0.0])
    >>> symplectic(state)
    tensor([[ 0.,  1.,  0.,  0.],
            [-1.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  1.],
            [ 0.,  0., -1.,  0.]])

    """
    block = torch.tensor([[0, 1], [-1, 0]], dtype=state.dtype, device=state.device)
    return torch.block_diag(*[block for _ in range(len(state) // 2)])


def first(xs:Iterable[Any]) -> Any:
    """
    Return first element

    Parameters
    ----------
    xs : Iterable[Any]
        xs

    Returns
    -------
    Any

    Examples
    --------
    >>> first([1, 2, 3, 4])
    1

    """
    x, *_ = xs
    return x


def last(xs:Iterable[Any]) -> Any:
    """
    Return last element

    Parameters
    ----------
    xs : Iterable[Any]
        xs

    Returns
    -------
    Any

    Examples
    --------
    >>> first([1, 2, 3, 4])
    4

    """
    *_, x = xs
    return x


def rest(xs:Iterable[Any]) -> Any:
    """
    Return all but last element

    Parameters
    ----------
    xs : Iterable[Any]
        xs

    Returns
    -------
    Any

    Examples
    --------
    >>> first([1, 2, 3, 4])
    [1, 2, 3]

    """
    *x, _ = xs
    return x


def most(xs:Iterable[Any]) -> Any:
    """
    Return all but first element

    Parameters
    ----------
    xs : Iterable[Any]
        xs

    Returns
    -------
    Any

    Examples
    --------
    >>> first([1, 2, 3, 4])
    [2, 3, 4]

    """
    _, *x = xs
    return x


@multimethod
def tolist(tensor:Tensor) -> list:
    """
    Convert input (gradtracking) tensor to list

    Note, emmits storage deprication warning

    Parameters
    ----------
    tensor : Tensor
        input tensor

    Returns
    -------
    list

    Examples
    --------
    >>> import torch
    >>> tolist(torch.tensor([0.0, 0.0, 0.0, 0.0]))
    [0.0, 0.0, 0.0, 0.0]

    """
    tensor = tensor.detach().cpu()
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        tensor = torch._C._functorch.get_unwrapped(tensor)
        return numpy.array(tensor.storage().tolist()).reshape(tensor.shape).tolist()
    return tensor.tolist()


@multimethod
def tolist(tensor:Array) -> list:
    """
    Convert input (gradtracking) tensor to list

    Note, emmits storage deprication warning

    Parameters
    ----------
    tensor : Array
        input tensor

    Returns
    -------
    list

    Examples
    --------
    >>> import numpy
    >>> tolist(numpy.array([0.0, 0.0, 0.0, 0.0]))
    [0.0, 0.0, 0.0, 0.0]

    """
    return tensor.tolist()
