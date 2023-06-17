"""
Index
-----

Derivative table representation utilities

"""

from typing import TypeAlias
from typing import Callable
from typing import Union

from multimethod import multimethod

import torch
from torch import Tensor

from .util import flatten
from .signature import signature
from .signature import get
from .signature import set


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
def index(dimension:int,
          order:int, *,
          dtype:torch.dtype=torch.int64,
          device:torch.device=torch.device('cpu')) -> Tensor:
    """
    Generate monomial index table with repetitions for a given dimension and order

    Note, output length is dimension**degree

    Parameters
    ----------
    dimension: int, positive
        monomial dimension (number of variables)
    order: int, non-negative
        derivative order (total monomial degree)
    dtype: torch.dtype, default=torch.int64
        data type
    device: torch.device, default=torch.device('cpu')
        data device

    Returns
    -------
    Tensor
        monomial index table with repetitions

    Examples
    --------
    >>> index(2, 3)
    tensor([[3, 0],
            [2, 1],
            [2, 1],
            [1, 2],
            [2, 1],
            [1, 2],
            [1, 2],
            [0, 3]])

    """
    if order == 0:
        return torch.zeros((1, dimension), dtype=dtype, device=device)

    if order == 1:
        return torch.eye(dimension, dtype=dtype, device=device)

    unit = index(dimension, 1, dtype=dtype, device=device)
    keys = index(dimension, order - 1, dtype=dtype, device=device)

    return torch.cat([keys + i for i in unit])


@multimethod
def index(dimension:tuple[int, ...],
          order:tuple[int, ...], *,
          dtype:torch.dtype=torch.int64,
          device:torch.device=torch.device('cpu')) -> Tensor:
    """
    Generate monomial index table with repetitions for given dimensions and corresponding orders

    Note, output length is product(dimension**degree)

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        monomial dimensions
    order: tuple[int, ...], non-negative
        derivative orders (total monomial degrees)
    dtype: torch.dtype, default=torch.int64
        data type
    device: torch.device, default=torch.device('cpu')
        data device

    Returns
    -------
    Tensor
        monomial index table with repetitions

    Example
    -------
    >>> index((2, 2), (3, 1))
    tensor([[3, 0, 1, 0],
            [3, 0, 0, 1],
            [2, 1, 1, 0],
            [2, 1, 0, 1],
            [2, 1, 1, 0],
            [2, 1, 0, 1],
            [1, 2, 1, 0],
            [1, 2, 0, 1],
            [2, 1, 1, 0],
            [2, 1, 0, 1],
            [1, 2, 1, 0],
            [1, 2, 0, 1],
            [1, 2, 1, 0],
            [1, 2, 0, 1],
            [0, 3, 1, 0],
            [0, 3, 0, 1]])

    """
    def merge(total:tuple, *table:tuple) -> tuple:
        x, *xs = table
        return tuple(merge(total + i, *xs) for i in x) if xs else tuple(list(total + i) for i in x)

    x, *xs = [tuple(index(*pair).tolist()) for pair in zip(dimension + (0, ), order + (0, ))]

    return torch.tensor([*flatten(tuple(merge(i, *xs) for i in x))], dtype=dtype, device=device)


@multimethod
def reduce(dimension:tuple[int, ...],
           signature:tuple[int, ...],
           tensor:Tensor) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], Tensor]:
    """
    Generate reduced representation of a given bottom element tensor

    Note, bottom element table is assumed to represent a scalar valued function

    Parameters
    ----------
    dimension: tuple[int]
        table derivative dimension
    signature: tuple[int]
        bottom element signature
    table: Table
        input derivative table

    Returns
    -------
    tuple[
        tuple[tuple[int, ...], ...],
        tuple[int, ...],
        Tensor]
        (sequence, shape, unique)

    Examples
    --------
    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.signature import get
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> def fn(x, y): x1, x2 = x; y1, = y; return x1 + x2 + (x1 + x2)*y1**2
    >>> t = derivative((1, 2), fn, x, y)
    >>> sequence, shape, unique = reduce((2, 1), (1, 2), get(t, (1, 2)))
    >>> sequence
    ((1, 0, 2), (0, 1, 2))
    >>> shape
    torch.Size([1, 1, 2])
    >>> unique
    {(1, 0, 2): tensor(2.), (0, 1, 2): tensor(2.)}

    """
    sequence = tuple(map(tuple, index(dimension, signature).tolist()))
    shape = tensor.shape
    unique = {}
    for key, value in zip(sequence, tensor.flatten()):
        if not key in unique:
            unique[key] = value
    return sequence, shape, unique


@multimethod
def reduce(dimension:tuple[int, ...],
           table:Table) -> tuple[
    dict[tuple[int, ...], tuple[tuple[int, ...], ...]],
    dict[tuple[int, ...], tuple[int, ...]],
    dict[tuple[int, ...], Tensor]]:
    """
    Generate reduced representation of a given derivative table

    Note, table is assumed to represent a scalar valued function

    Parameters
    ----------
    dimension: tuple[int]
        table derivative dimension
    table: Table
        input derivative table

    Returns
    -------
        tuple[
            dict[tuple[int, ...], tuple[tuple[int, ...], ...]],
            dict[tuple[int, ...], tuple[int, ...]],
            dict[tuple[int, ...], Tensor]
        ]
        (sequence, shape, unique)

    Examples
    --------
    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> def fn(x, y): x1, x2 = x; y1, = y; return x1 + x2 + (x1 + x2)*y1**2
    >>> t = derivative((1, 2), fn, x, y)
    >>> t
    [[tensor(0.), tensor([0.]), tensor([[0.]])],
     [tensor([1.000000e+00, 1.000000e+00]),
      tensor([[0., 0.]]),
      tensor([[[2.000000e+00, 2.000000e+00]]])]]
    >>> sequence, shape, unique = reduce((2, 1), t)
    >>> sequence
    {(0, 0): ((0, 0, 0),),
     (0, 1): ((0, 0, 1),),
     (0, 2): ((0, 0, 2),),
     (1, 0): ((1, 0, 0), (0, 1, 0)),
     (1, 1): ((1, 0, 1), (0, 1, 1)),
     (1, 2): ((1, 0, 2), (0, 1, 2))}
    >>> shape
    {(0, 0): torch.Size([]),
     (0, 1): torch.Size([1]),
     (0, 2): torch.Size([1, 1]),
     (1, 0): torch.Size([2]),
     (1, 1): torch.Size([1, 2]),
     (1, 2): torch.Size([1, 1, 2])}
    >>> unique
    {(0, 0, 0): tensor(0.),
     (0, 0, 1): tensor(0.),
     (0, 0, 2): tensor(0.),
     (1, 0, 0): tensor(1.000000e+00),
     (0, 1, 0): tensor(1.000000e+00),
     (1, 0, 1): tensor(0.),
     (0, 1, 1): tensor(0.),
     (1, 0, 2): tensor(2.000000e+00),
     (0, 1, 2): tensor(2.000000e+00)}

    """
    sequence, shape, unique = {}, {}, {}
    for i in signature(table):
        sequence[i] = tuple(map(tuple, index(dimension, i).tolist()))
        tensor = get(table, i)
        shape[i] = tensor.shape
        for key, value in zip(sequence[i], tensor.flatten()):
            if not key in unique:
                unique[key] = value
    return sequence, shape, unique


def bottom(sequence:tuple[tuple[int, ...], ...],
           shape:tuple[int],
           unique:dict[tuple[int, ...], Tensor]) -> Tensor:
    """
    Generate bottom derivative table element from reduced data

    Note, bottom element table is assumed to represent a scalar valued function

    Parameters
    ----------
    sequence: tuple[tuple[int, ...], ...]
        sequence of monomial indices with repetitions (see index function)
    shape: tuple[int]
        output tensor shape
    unique: dict[tuple[int, ...], Tensor]
        unique values

    Returns
    -------
    Tensor

    Examples
    --------
    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.signature import get
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> def fn(x, y): x1, x2 = x; y1, = y; return x1 + x2 + (x1 + x2)*y1**2
    >>> t = derivative((1, 2), fn, x, y)
    >>> sequence, shape, unique = reduce((2, 1), t)
    >>> get(t, (1, 2))
    tensor([[[2.000000e+00, 2.000000e+00]]])
    >>> bottom(sequence[(1, 2)], shape[(1, 2)], unique)
    tensor([[[2.000000e+00, 2.000000e+00]]])
    
    """
    return torch.stack([unique[index] for index in sequence]).reshape(shape)


def build(table:Table,
          sequence:dict[tuple[int, ...], tuple[tuple[int, ...], ...]],
          shape:dict[tuple[int, ...], tuple[int, ...]],
          unique:dict[tuple[int, ...], Tensor]) -> None:
    """
    Build derivative table representation from a given reduced representation

    Note, table is assumed to represent a scalar valued function
    Note, modify input container

    Parameters
    ----------
    table: Table
        container
    sequence: dict[tuple[int, ...], tuple[tuple[int, ...], ...]]
        sequence of monomial indices with repetitions (see index function)
    shape: dict[tuple[int, ...], tuple[int, ...]]
        output tensor shape
    unique: dict[tuple[int, ...], Tensor]
        unique values

    Returns
    -------
    None

    Examples
    --------
    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> def fn(x, y): x1, x2 = x; y1, = y; return x1 + x2 + (x1 + x2)*y1**2
    >>> t = derivative((1, 2), fn, x, y)
    >>> t
    [[tensor(0.), tensor([0.]), tensor([[0.]])],
     [tensor([1.000000e+00, 1.000000e+00]),
      tensor([[0., 0.]]),
      tensor([[[2.000000e+00, 2.000000e+00]]])]]
    >>> s = derivative((1, 2), lambda x, y: torch.zeros_like(x).sum(), x, y)
    >>> build(s, *reduce((2, 1), t))
    >>> s
    [[tensor(0.), tensor([0.]), tensor([[0.]])],
     [tensor([1.000000e+00, 1.000000e+00]),
      tensor([[0., 0.]]),
      tensor([[[2.000000e+00, 2.000000e+00]]])]]

    """
    for i in signature(table):
        set(table, i, bottom(sequence[i], shape[i], unique))
