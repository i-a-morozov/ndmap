"""
Index
-----

Derivative table representation utilities

"""

from typing import TypeAlias
from typing import Callable
from typing import Union
from typing import Optional

from multimethod import multimethod

import numpy
from numpy import ndarray as Array

import torch
from torch import Tensor

from ndmap.util import tolist
from ndmap.signature import signature
from ndmap.signature import get
from ndmap.signature import set

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
          order:int) -> Array:
    """
    Generate monomial index table with repetitions for a given dimension and order

    Note, output length is dimension**degree

    Parameters
    ----------
    dimension: int, positive
        monomial dimension (number of variables)
    order: int, non-negative
        derivative order (total monomial degree)

    Returns
    -------
    Array
        monomial index table with repetitions

    Examples
    --------
    >>> index(2, 3)
    array([[3, 0],
           [2, 1],
           [2, 1],
           [1, 2],
           [2, 1],
           [1, 2],
           [1, 2],
           [0, 3]])

    """
    if order == 0:
        return numpy.zeros((1, dimension), dtype=numpy.int64)

    if order == 1:
        return numpy.eye(dimension, dtype=numpy.int64)

    unit = index(dimension, 1)
    keys = index(dimension, order - 1)

    return numpy.concatenate([keys + i for i in unit])


@multimethod
def index(dimension:tuple[int, ...],
          order:tuple[int, ...]) -> Array:
    """
    Generate monomial index table with repetitions for given dimensions and corresponding orders

    Note, output length is product(dimension**degree)

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        monomial dimensions
    order: tuple[int, ...], non-negative
        derivative orders (total monomial degrees)

    Returns
    -------
    Array
        monomial index table with repetitions

    Example
    -------
    >>> index((2, 2), (3, 1))
    array([[3, 0, 1, 0],
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
    def merge(head:Array, *tail:Array) -> Array:
        x, *xs = tail
        if not len(xs):
            return numpy.vstack([numpy.concatenate([head, i]) for i in x])
        return numpy.concatenate([merge(numpy.concatenate([head, i]), *xs)for i in x])
    dimension, order = dimension + (0, ), order + (0, )
    x, *xs = [index(*pair) for pair in zip(dimension, order)]
    return numpy.vstack([merge(i, *xs) for i in x])


@multimethod
def reduce(dimension:tuple[int, ...],
           signature:tuple[int, ...],
           tensor:Tensor, *,
           sequence:Optional[Tensor]=None) -> tuple[tuple, tuple, dict]:
    """
    Generate reduced representation of a given bottom element tensor

    Note, bottom element table is assumed to represent a mapping or a scalar

    Parameters
    ----------
    dimension: tuple[int, ...]
        table derivative dimension
    signature: tuple[int, ...]
        bottom element signature
    table: Table
        input derivative table
    sequence: Optional[Tensor]
        precomputed sequence

    Returns
    -------
    tuple[tuple, tuple, dict]
        (sequence, shape, unique)

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> from ndmap.signature import get
    >>> def fn(x, y): x1, x2 = x; y1, = y; return torch.stack([x1*y1 + x2, x2**2]).sum()
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> t = derivative((2, 1), fn, x, y)
    >>> t
    [[tensor(0.), tensor([0.])],
    [tensor([0., 1.]), tensor([[1., 0.]])],
    [tensor([[0., 0.],
            [0., 2.]]),
    tensor([[[0., 0.],
            [0., 0.]]])]]
    >>> sequence, shape, unique = reduce((2, 1), (2, 0), get(t, (2, 0)))
    >>> sequence
    ((2, 0, 0), (1, 1, 0), (1, 1, 0), (0, 2, 0))
    >>> shape
    torch.Size([2, 2])
    >>> unique
    {(2, 0, 0): tensor(0.), (1, 1, 0): tensor(0.), (0, 2, 0): tensor(2.)}

    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> from ndmap.signature import get
    >>> def fn(x, y): x1, x2 = x; y1, = y; return torch.stack([x1*y1 + x2, x2**2])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> t = derivative((2, 1), fn, x, y)
    >>> sequence, shape, unique = reduce((2, 1), (2, 0), get(t, (2, 0)))
    >>> sequence
    ((2, 0, 0), (1, 1, 0), (1, 1, 0), (0, 2, 0))
    >>> shape
    torch.Size([2, 2, 2])
    >>> unique
    {(2, 0, 0): tensor([0., 0.]),
    (1, 1, 0): tensor([0., 0.]),
    (0, 2, 0): tensor([0., 2.])}


    """
    sequence = index(dimension, signature) if sequence is None else sequence
    shape = tuple(tensor.shape)
    array = tuple(reversed(range(len(shape))))
    if array:
        tensor = tensor.permute(array)
    tensor = tensor.flatten().reshape(len(sequence), -1).squeeze(-1)
    unique = {}
    for key, value in zip(tuple(map(tuple, tolist(sequence))), tensor):
        if key not in unique:
            unique[key] = value
    length, *size = tensor.shape
    size = shape[len(size):] if size else shape
    array = tuple(reversed(range(len(size))))
    table = numpy.arange(0, length)
    table = table.reshape(tuple(reversed(tuple(size))))
    table = numpy.transpose(table, array).flatten()
    sequence = tuple(map(tuple, tolist(sequence[table])))
    return sequence, shape, unique


@multimethod
def reduce(dimension:tuple[int, ...],
           table:Table) -> tuple[dict, dict, dict]:
    """
    Generate reduced representation of a given derivative table

    Note, table is assumed to represent a mapping or a scalar

    Parameters
    ----------
    dimension: tuple[int, ...]
        table derivative dimension
    table: Table
        input derivative table

    Returns
    -------
    tuple[dict, dict, dict]
        (sequence, shape, unique)

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> from ndmap.signature import get
    >>> def fn(x, y): x1, x2 = x; y1, = y; return torch.stack([x1*y1 + x2, x2**2]).sum()
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> t = derivative((2, 1), fn, x, y)
    >>> t
    [[tensor(0.), tensor([0.])],
    [tensor([0., 1.]), tensor([[1., 0.]])],
    [tensor([[0., 0.],
            [0., 2.]]),
    tensor([[[0., 0.],
            [0., 0.]]])]]
    >>> sequence, shape, unique = reduce((2, 1), t)
    >>> sequence
    {(0, 0): ((0, 0, 0),),
    (0, 1): ((0, 0, 1),),
    (1, 0): ((1, 0, 0), (0, 1, 0)),
    (1, 1): ((1, 0, 1), (0, 1, 1)),
    (2, 0): ((2, 0, 0), (1, 1, 0), (1, 1, 0), (0, 2, 0)),
    (2, 1): ((2, 0, 1), (1, 1, 1), (1, 1, 1), (0, 2, 1))}
    >>> shape
    {(0, 0): torch.Size([]),
    (0, 1): torch.Size([1]),
    (1, 0): torch.Size([2]),
    (1, 1): torch.Size([1, 2]),
    (2, 0): torch.Size([2, 2]),
    (2, 1): torch.Size([1, 2, 2])}
    >>> unique
    {(0, 0, 0): tensor(0.),
    (0, 0, 1): tensor(0.),
    (1, 0, 0): tensor(0.),
    (0, 1, 0): tensor(1.),
    (1, 0, 1): tensor(1.),
    (0, 1, 1): tensor(0.),
    (2, 0, 0): tensor(0.),
    (1, 1, 0): tensor(0.),
    (0, 2, 0): tensor(2.),
    (2, 0, 1): tensor(0.),
    (1, 1, 1): tensor(0.),
    (0, 2, 1): tensor(0.)}

    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y): x1, x2 = x; y1, = y; return torch.stack([x1*y1 + x2, x2**2])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> t = derivative((2, 1), fn, x, y)
    >>> sequence, shape, unique = reduce((2, 1), t)
    >>> sequence
    {(0, 0): ((0, 0, 0),),
     (0, 1): ((0, 0, 1),),
     (1, 0): ((1, 0, 0), (0, 1, 0)),
     (1, 1): ((1, 0, 1), (0, 1, 1)),
     (2, 0): ((2, 0, 0), (1, 1, 0), (1, 1, 0), (0, 2, 0)),
     (2, 1): ((2, 0, 1), (1, 1, 1), (1, 1, 1), (0, 2, 1))}
    >>> shape
    {(0, 0): torch.Size([2]),
     (0, 1): torch.Size([2, 1]),
     (1, 0): torch.Size([2, 2]),
     (1, 1): torch.Size([2, 1, 2]),
     (2, 0): torch.Size([2, 2, 2]),
     (2, 1): torch.Size([2, 1, 2, 2])}
    >>> unique
    {(0, 0, 0): tensor([0., 0.]),
     (0, 0, 1): tensor([0., 0.]),
     (1, 0, 0): tensor([0., 0.]),
     (0, 1, 0): tensor([1., 0.]),
     (1, 0, 1): tensor([1., 0.]),
     (0, 1, 1): tensor([0., 0.]),
     (2, 0, 0): tensor([0., 0.]),
     (1, 1, 0): tensor([0., 0.]),
     (0, 2, 0): tensor([0., 2.]),
     (2, 0, 1): tensor([0., 0.]),
     (1, 1, 1): tensor([0., 0.]),
     (0, 2, 1): tensor([0., 0.])}

    """
    sequence, shape, unique = {}, {}, {}
    for i in signature(table):
        sequence[i], shape[i], local = reduce(dimension, i, get(table, i))
        unique.update(local)
    return sequence, shape, unique


@multimethod
def build(sequence:tuple,
          shape:tuple,
          unique:dict) -> Tensor:
    """
    Generate bottom derivative table element from reduced data

    Note, bottom element table is assumed to represent a scalar valued function

    Parameters
    ----------
    sequence: tuple
        sequence of monomial indices with repetitions (see index function)
    shape: tuple
        output tensor shape
    unique: dict
        unique values

    Returns
    -------
    Tensor

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> from ndmap.signature import get
    >>> def fn(x, y): x1, x2 = x; y1, = y; return torch.stack([x1*y1 + x2, x2**2]).sum()
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> t = derivative((2, 1), fn, x, y)
    >>> build(*reduce((2, 1), (2, 0), get(t, (2, 0))))
    tensor([[0., 0.],
            [0., 2.]])
    >>> get(t, (2, 0))
    tensor([[0., 0.],
            [0., 2.]])

    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> from ndmap.signature import get
    >>> def fn(x, y): x1, x2 = x; y1, = y; return torch.stack([x1*y1 + x2, x2**2])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> t = derivative((2, 1), fn, x, y)
    >>> build(*reduce((2, 1), (2, 0), get(t, (2, 0))))
    tensor([[[0., 0.],
             [0., 0.]],

            [[0., 0.],
             [0., 2.]]])
    >>> get(t, (2, 0))
    tensor([[[0., 0.],
             [0., 0.]],

            [[0., 0.],
             [0., 2.]]])

    """
    return torch.stack([unique[index] for index in sequence]).swapaxes(0, -1).reshape(shape)


@multimethod
def build(table:Table,
          sequence:dict,
          shape:dict,
          unique:dict) -> None:
    """
    Build derivative table representation from a given reduced representation

    Note, table is assumed to represent a mapping or a scalar valued function
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
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y): x1, x2 = x; y1, = y; return torch.stack([x1*y1 + x2, x2**2]).sum()
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> t = derivative((2, 2), fn, x, y)
    >>> s = derivative((2, 2), lambda x, y: x.sum(), x, y)
    >>> build(s, *reduce((2, 1), t))
    >>> s
    [[tensor(0.), tensor([0.]), tensor([[0.]])],
     [tensor([0., 1.]), tensor([[1., 0.]]), tensor([[[0., 0.]]])],
     [tensor([[0., 0.],
              [0., 2.]]),
      tensor([[[0., 0.],
               [0., 0.]]]),
      tensor([[[[0., 0.],
                [0., 0.]]]])]]

    >>> import torch
    >>> from ndmap.util import equal
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y): x1, x2 = x; y1, = y; return torch.stack([x1*y1 + x2, x2**2])
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([0.0])
    >>> t = derivative((2, 1), fn, x, y)
    >>> t = derivative((2, 2), fn, x, y)
    >>> s = derivative((2, 2), lambda x, y: x, x, y)
    >>> build(s, *reduce((2, 1), t))
    >>> equal(t, s)
    True

    """
    for i in signature(table):
        set(table, i, build(sequence[i], shape[i], unique))


@multimethod
def _index(dimension:int,
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
    >>> _index(2, 3)
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

    unit = _index(dimension, 1, dtype=dtype, device=device)
    keys = _index(dimension, order - 1, dtype=dtype, device=device)

    return torch.cat([keys + i for i in unit])


@multimethod
def _index(dimension:tuple[int, ...],
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
    >>> _index((2, 2), (3, 1))
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
    def merge(head:Tensor, *tail:Tensor) -> Tensor:
        x, *xs = tail
        if not len(xs):
            return torch.vstack([torch.cat([head, i]) for i in x])
        return torch.cat([merge(torch.cat([head, i]), *xs)for i in x])
    dimension, order = dimension + (0, ), order + (0, )
    x, *xs = [_index(*pair, dtype=dtype, device=device) for pair in zip(dimension, order)]
    return torch.vstack([merge(i, *xs) for i in x])
