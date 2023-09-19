"""
Series
------

Series representation of derivative table

"""

from math import factorial
from math import prod

from typing import TypeAlias
from typing import Callable
from typing import Optional
from typing import Union

from multimethod import multimethod

import torch
from torch import Tensor

from ndmap.util import flatten
from ndmap.util import tolist
from ndmap.derivative import derivative
from ndmap.signature import signature
from ndmap.index import index


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


@multimethod
def series(dimension:tuple[int, ...],
           order:tuple[int, ...],
           table:Table) -> Series:
    """
    Generate series representation from a given derivative table representation

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        dimensions
    order: tuple[int, ...], non-negative
        derivative orders
    table: Table
        derivative table representation

    Returns
    -------
    Series

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return torch.stack([x1*(1 + y1), x2*(1 + y2)])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((1, 1), fn, x, y)
    >>> series((2, 2), (1, 1), t)
    {(0, 0, 0, 0): tensor([0., 0.]),
     (0, 0, 1, 0): tensor([0., 0.]),
     (0, 0, 0, 1): tensor([0., 0.]),
     (1, 0, 0, 0): tensor([1., 0.]),
     (0, 1, 0, 0): tensor([0., 1.]),
     (1, 0, 1, 0): tensor([1., 0.]),
     (1, 0, 0, 1): tensor([0., 0.]),
     (0, 1, 1, 0): tensor([0., 0.]),
     (0, 1, 0, 1): tensor([0., 1.])}

    """
    series = {}

    for (count, factor), array in zip(signature(table, factor=True), flatten(table, target=list)):
        if not all(i <= j for i, j in zip(count, order)):
            continue
        count = tolist(index(dimension, count))
        shape = tuple(reversed(range(len(array.shape))))
        array = factor*(array.permute(shape) if shape else array)
        array = array.flatten().reshape(len(count), -1).clone()
        for key, value in zip(count, array):
            key = tuple(key)
            if key not in series:
                series[key] = value
            else:
                series[key] = series[key] + value

    return series


@multimethod
def series(index:tuple[int, ...],
           function:Callable,
           *args:tuple,
           jacobian:Optional[Callable]=None) -> Series:
    """
    Generate series representation of a given input function upto a given monomial index

    c(i, j, k, ...) * x**i * y**j * z**k * ... => {..., (i, j, k, ...) : c(i, j, k, ...), ...}

    Note, input function arguments are expected to be scalar tensors

    Parameters
    ----------
    index: tuple[int, ...], non-negative
        monomial index, (i, j, k, ...)
    function: Callable,
        input function
    *args: tuple
        input function arguments
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Series

    Examples
    --------
    >>> import torch
    >>> from ndmap.util import curry_apply
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return torch.stack([x1*(1 + y1), x2*(1 + y2)])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> series((1, 1, 1, 1), curry_apply(fn, (2, 2)), *x, *y)
    {(0, 0, 0, 0): tensor([0., 0.]),
     (0, 0, 0, 1): tensor([0., 0.]),
     (0, 0, 1, 0): tensor([0., 0.]),
     (0, 0, 1, 1): tensor([0., 0.]),
     (0, 1, 0, 0): tensor([0., 1.]),
     (0, 1, 0, 1): tensor([0., 1.]),
     (0, 1, 1, 0): tensor([0., 0.]),
     (0, 1, 1, 1): tensor([0., 0.]),
     (1, 0, 0, 0): tensor([1., 0.]),
     (1, 0, 0, 1): tensor([0., 0.]),
     (1, 0, 1, 0): tensor([1., 0.]),
     (1, 0, 1, 1): tensor([0., 0.]),
     (1, 1, 0, 0): tensor([0., 0.]),
     (1, 1, 0, 1): tensor([0., 0.]),
     (1, 1, 1, 0): tensor([0., 0.]),
     (1, 1, 1, 1): tensor([0., 0.])}

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian
    return series((1, ) * len(args),
                  index,
                  derivative(index, function, *args, intermediate=True, jacobian=jacobian))


@multimethod
def series(index:list[tuple[int, ...]],
           function:Callable,
           *args:tuple,
           jacobian:Optional[Callable]=None) -> Series:
    """
    Generate series representation of a given input function for a given set of monomial indices

    c(i, j, k, ...) * x**i * y**j * z**k * ... => {..., (i, j, k, ...) : c(i, j, k, ...), ...}

    Note, input function arguments are expected to be scalar tensors

    Parameters
    ----------
    index: list[tuple[int, ...]], non-negative
        list of monomial indices, [..., (i, j, k, ...), ...]
    function: Callable,
        input function
    *args: tuple
        input function arguments
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Series

    Examples
    --------
    >>> import torch
    >>> from ndmap.util import curry_apply
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return torch.stack([x1*(1 + y1), x2*(1 + y2)])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> series([(1,0,0,0), (0,1,0,0), (1,0,1,0), (0,1,0,1)], curry_apply(fn, (2, 2)), *x, *y)
    {(1, 0, 0, 0): tensor([1., 0.]),
     (0, 1, 0, 0): tensor([0., 1.]),
     (1, 0, 1, 0): tensor([1., 0.]),
     (0, 1, 0, 1): tensor([0., 1.])}

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    def factor(*index:tuple[int]) -> float:
        return 1.0 / prod(map(factorial, index))

    return {i: factor(*i) * derivative(i,
                                       function,
                                       *args,
                                       intermediate=False,
                                       jacobian=jacobian) for i in index}


@multimethod
def series(index:list[tuple[int, ...]],
           function:Callable,
           point:Point,
           *pars:tuple,
           jacobian:Optional[Callable]=None) -> Series:
    """ Generate series representation of a given function for a given set of monomial indices """
    return series(index, function, *torch.cat(point), *pars, jacobian=jacobian)


@multimethod
def series(index:list[tuple[int, ...]],
           function:Callable,
           state:State,
           knobs:Knobs,
           *pars:tuple,
           jacobian:Optional[Callable]=None) -> Series:
    """ Generate series representation of a given function for a given set of monomial indices """
    return series(index, function, *state, *torch.cat(knobs), *pars, jacobian=jacobian)


def merge(probe:Series,
          other:Series) -> Series:
    """
    Merge (sum) series

    Parameters
    ----------
    probe, other: Series
        input series to merge

    Returns
    -------
    Series

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return torch.stack([x1*(1 + y1), x2*(1 + y2)])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((1, 1), fn, x, y)
    >>> s = series((2, 2), (1, 1), t)
    >>> merge(s, s)
    {(0, 0, 0, 0): tensor([0., 0.]),
     (0, 0, 1, 0): tensor([0., 0.]),
     (0, 0, 0, 1): tensor([0., 0.]),
     (1, 0, 0, 0): tensor([2., 0.]),
     (0, 1, 0, 0): tensor([0., 2.]),
     (1, 0, 1, 0): tensor([2., 0.]),
     (1, 0, 0, 1): tensor([0., 0.]),
     (0, 1, 1, 0): tensor([0., 0.]),
     (0, 1, 0, 1): tensor([0., 2.])}

    """
    total = {key: value.clone() for key, value in probe.items()}
    for key, value in other.items():
        if key in total:
            total[key] += value
        else:
            total[key]  = value
    return total


def clean(probe:Series, *,
          epsilon:float=2.5E-16) -> Series:
    """
    Clean series

    Parameters
    ----------
    probe: Series
        input series to clean
    epsilon: float, non-negative
        clean epsilon

    Returns
    -------
    Series

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return torch.stack([x1*(1 + y1), x2*(1 + y2)])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((1, 1), fn, x, y)
    >>> s = series((2, 2), (1, 1), t)
    >>> clean(s)
    {(1, 0, 0, 0): tensor([1., 0.]),
     (0, 1, 0, 0): tensor([0., 1.]),
     (1, 0, 1, 0): tensor([1., 0.]),
     (0, 1, 0, 1): tensor([0., 1.])}

    """
    return {key: value for key, value in probe.items() if torch.any(value.abs() > epsilon)}


def fetch(probe:Series,
          index:list[tuple[int, ...]]) -> Series:
    """
    Fetch series

    Parameters
    ----------
    probe: Series
        input series
    index: list[tuple[int, ...]], non-negative
        list of monomial indices, [..., (i, j, k, ...), ...]

    Returns
    -------
    Series

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return torch.stack([x1*(1 + y1), x2*(1 + y2)])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((1, 1), fn, x, y)
    >>> s = series((2, 2), (1, 1), t)
    >>> fetch(s, [(1, 0, 0, 0), (1, 0, 1, 0)])
    {(1, 0, 0, 0): tensor([1., 0.]), (1, 0, 1, 0): tensor([1., 0.])}

    """
    return {key: value for key, value in probe.items() if key in index}


def split(probe:Series) -> list[Series]:
    """
    (series operation) Split series

    Note, coefficient values are assumed to be vector tensors

    Parameters
    ----------
    probe: Series
        input series

    Returns
    -------
    list[Series]

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return torch.stack([x1*(1 + y1), x2*(1 + y2)])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((1, 1), fn, x, y)
    >>> s = series((2, 2), (1, 1), t)
    >>> split(s)
    [{(0, 0, 0, 0): tensor(0.),
      (0, 0, 1, 0): tensor(0.),
      (0, 0, 0, 1): tensor(0.),
      (1, 0, 0, 0): tensor(1.),
      (0, 1, 0, 0): tensor(0.),
      (1, 0, 1, 0): tensor(1.),
      (1, 0, 0, 1): tensor(0.),
      (0, 1, 1, 0): tensor(0.),
      (0, 1, 0, 1): tensor(0.)},
     {(0, 0, 0, 0): tensor(0.),
      (0, 0, 1, 0): tensor(0.),
      (0, 0, 0, 1): tensor(0.),
      (1, 0, 0, 0): tensor(0.),
      (0, 1, 0, 0): tensor(1.),
      (1, 0, 1, 0): tensor(0.),
      (1, 0, 0, 1): tensor(0.),
      (0, 1, 1, 0): tensor(0.),
      (0, 1, 0, 1): tensor(1.)}]

    """
    return [dict(zip(probe.keys(), value)) for value in torch.stack([*probe.values()]).T]
