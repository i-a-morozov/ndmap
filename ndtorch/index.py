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
