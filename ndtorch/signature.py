"""
Signature
---------

Derivative table representation utilities

"""
from math import factorial

from typing import TypeAlias
from typing import Callable
from typing import Union

from multimethod import multimethod

from torch import Tensor

from .util import flatten

Mapping  : TypeAlias = Callable
Point    : TypeAlias = list[Tensor]
Delta    : TypeAlias = list[Tensor]
State    : TypeAlias = Tensor
Knobs    : TypeAlias = list[Tensor]
Table    : TypeAlias = list
Series   : TypeAlias = dict[tuple[int, ...], Tensor]
Signature: TypeAlias = Union[list[tuple[int, ...]], list[tuple[tuple[int, ...], float]]]


@multimethod
def signature(table:Table, *,
              factor:bool=False) -> Signature:
    """
    Compute derivative table bottom elements signatures

    Note, signature elements corresponds to the bottom elements of a flattened derivative table
    Bottom element signature is a tuple integers, derivative orders with respect to each tensor
    Optionaly return elements multiplication factors
    Given a signature (n, m, ...), corresponding multiplication factor is 1/n! * 1/m! * ...

    Parameters
    ----------
    table: Table
        input derivative table representation
    fator: bool, default=True
        flag to return elements multipliation factors

    Returns
    -------
    Signature
        bottom table elements signatures

    Examples
    --------
    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((2, 1), fn, x, y)
    >>> signature(t)
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    >>> signature(t, factor=True)
    [((0, 0), 1.0),
     ((0, 1), 1.0),
     ((1, 0), 1.0),
     ((1, 1), 1.0),
     ((2, 0), 0.5),
     ((2, 1), 0.5)]
    
    """
    array = [signature([i], subtable, factor=factor) for i, subtable in enumerate(table)]
    return [*flatten(array, target=list)]


@multimethod
def signature(index:list[int],
              table:Table, *,
              factor:bool=False):
    """ (auxiliary) """
    return [signature(index + [i], subtable, factor=factor) for i, subtable in enumerate(table)]


@multimethod
def signature(index:list[int],
              table:Tensor, *,
              factor:bool=False):
    """ (auxiliary) """
    value = 1.0
    for count in index:
        value *= 1.0/factorial(count)
    return tuple(index) if not factor else (tuple(index), value)


def get(table:Table, index:tuple[int, ...]) -> Union[Tensor, Table]:
    """
    Get derivative table element at a given (bottom) element signature

    Note, index can correspond to a bottom element or a subtable

    Parameters
    ----------
    table: Table
        input derivative table representation
    index: tuple[int, ...]
        element signature

    Returns
    -------
    Union[Tensor, Table]
        element value

    Examples
    --------
    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((2, 1), fn, x, y)
    >>> get(t, (1, 1))
    tensor([[1., 1.],
        [1., 1.]])

    """
    if isinstance(index, int):
        return table[index]

    *ns, n = index
    for i in ns:
        table = table[i]
    return table[n]


def set(table:Table, index:tuple[int, ...], value:Union[Tensor, Table]) -> None:
    """
    Set derivative table element at a given (bottom) element signature.

    Note, index can correspond to a bottom element or a subtable

    Parameters
    ----------
    table: Table
        input derivative table representation
    index: tuple[int, ...]
        element signature
    value: Union[Tensor, Table]
        element value

    Returns
    -------
    None

    Examples
    --------
    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((2, 1), fn, x, y)
    >>> set(t, (1, 1), 1 + get(t, (1, 1)))
    >>> get(t, (1, 1))
    tensor([[2., 2.],
            [2., 2.]])

    """
    if isinstance(index, int):
        table[index] = value
        return

    *ns, n = index
    for i in ns:
        table = table[i]
    table[n] = value


def apply(table:Table, index:tuple[int, ...], function:Callable) -> None:
    """
    Apply function (modifies element at index).

    Note, index can correspond to a bottom element or a subtable

    Parameters
    ----------
    table: Table
        input derivative table representation
    index: tuple[int, ...]
        element signature
    function: Callable
        function to apply

    Returns
    -------
    None

    Examples
    --------
    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((2, 1), fn, x, y)
    >>> apply(t, (1, 1), torch.log)
    >>> get(t, (1, 1))

    """
    value = get(table, index)
    set(table, index, function(value))
