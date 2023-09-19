"""
Propagate
---------

Propagate table and series representations

"""

from typing import TypeAlias
from typing import Callable
from typing import Optional
from typing import Union

from multimethod import multimethod

import torch
from torch import Tensor

from ndmap.util import flatten
from ndmap.derivative import derivative
from ndmap.signature import set
from ndmap.signature import get
from ndmap.series import series
from ndmap.evaluate import evaluate


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


def identity(order:tuple[int, ...],
             point:Point, *,
             flag:bool=False,
             parametric:Optional[Table]=None,
             jacobian:Optional[Callable]=None) -> Union[Table, Series]:
    """
    Generate identity derivative table or identity series

    Note, identity table or series represent an identity mapping

    Parameters
    ----------
    order: tuple[int, ...], non-negative
        maximum derivative orders
    point: Point
        evaluation point
    flag: bool, default=False
        flag to return identity series instead of table
    parametric: Optional[Table]
        optional parametric table
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Union[Table, Series]
        identity derivative table or series

    Examples
    --------
    >>> import torch
    >>> state = torch.tensor([0.0, 0.0])
    >>> knobs = [torch.tensor([0.0, 0.0])]
    >>> point = [state, *knobs]
    >>> identity((1, 1), point)
    [[tensor([0., 0.]),
    tensor([[0., 0.],
            [0., 0.]])],
    [tensor([[1., 0.],
            [0., 1.]]),
    tensor([[[0., 0.],
            [0., 0.]],
    
            [[0., 0.],
            [0., 0.]]])]]
    >>> identity((1, 1), point, flag=True)
    {(0, 0, 0, 0): tensor([0., 0.]),
    (0, 0, 1, 0): tensor([0., 0.]),
    (0, 0, 0, 1): tensor([0., 0.]),
    (1, 0, 0, 0): tensor([1., 0.]),
    (0, 1, 0, 0): tensor([0., 1.]),
    (1, 0, 1, 0): tensor([0., 0.]),
    (1, 0, 0, 1): tensor([0., 0.]),
    (0, 1, 1, 0): tensor([0., 0.]),
    (0, 1, 0, 1): tensor([0., 0.])}

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    table = derivative(order, lambda x, *xs: x, point, intermediate=True, jacobian=jacobian)

    if parametric is not None:
        set(table, (0, ),  get(parametric, (0, )))

    if not flag:
        return table

    return series(tuple(map(len, point)), order, table)


@multimethod
def propagate(dimension:tuple[int, ...],
              order:tuple[int, ...],
              data:Table,
              knobs:Knobs,
              mapping:Mapping,
              *pars:tuple,
              intermediate:bool=True,
              jacobian:Optional[Callable]=None) -> Union[Table, Tensor]:
    """
    Propagate derivative table representation through a given mapping

    Note, can propagate through a scalar observable

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        dimensions
    order: tuple[int, ...], non-negative
        maximum derivative orders
    data: Table
        input derivative table
    knobs: Knobs
        input parametric variables
    mapping: Mapping
        input mapping
    *pars: tuple
        additional mapping fixed arguments
    intermediate: bool, default=True
        flag to return intermediate derivatives
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Union[Table, Tensor]

    Examples
    --------
    >>> import torch
    >>> from ndmap.util import flatten
    >>> from ndmap.derivative import derivative
    >>> state = torch.tensor([0.0, 0.0])
    >>> knobs = torch.tensor([1.0, 1.0])
    >>> def fn(state, knobs):
    ...     x1, x2 = state
    ...     y1, y2 = knobs
    ...     return torch.stack([x1 + x2*y1, x2 + x1*y2])
    >>> def gn(state, knobs):
    ...     x1, x2 = state
    ...     y1, y2 = knobs
    ...     return torch.stack([y1 + y2*x1, y2 + y1*x2])
    >>> u = derivative((2, 2), lambda state, knobs: gn(fn(state, knobs), knobs), state, knobs)
    >>> v = identity((2, 2), [state, knobs])
    >>> v = propagate((2, 2), (2, 2), v, [knobs], fn)
    >>> v = propagate((2, 2), (2, 2), v, [knobs], gn)
    >>> all(torch.allclose(x, y) for x, y in zip(*map(lambda t: flatten(t, target=list), (u, v))))
    True

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    def auxiliary(*args) -> Tensor:
        state, *args = args
        state = evaluate(data, [state, *args])
        args = [arg + knob for arg, knob in zip(args, knobs)]
        return mapping(state, *args, *pars)

    value, *_ = flatten(data, target=list)
    delta = [torch.zeros(i, dtype=value.dtype, device=value.device) for i in dimension]

    return derivative(order, auxiliary, delta, intermediate=intermediate, jacobian=jacobian)


@multimethod
def propagate(dimension:tuple[int, ...],
              order:tuple[int, ...],
              data:Series,
              knobs:Knobs,
              mapping:Mapping,
              *pars:tuple,
              epsilon:Optional[float]=None,
              jacobian:Optional[Callable]=None) -> Series:
    """
    Propagate series representation through a given mapping

    Note, input series are expected to contain all indices

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        dimensions
    order: tuple[int, ...], non-negative
        maximum derivative orders
    data: Series
        input series
    knobs: Knobs
        input parametric variables
    mapping: Mapping
        input mapping
    *pars: tuple
        additional mapping fixed arguments
    epsilon: Optional[float], non-negative, default=None
        fast series evaluation / tolerance epsilon
    intermediate: bool, default=True
        flag to return intermediate derivatives
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Series

    Examples
    --------
    >>> import torch
    >>> from ndmap.util import flatten
    >>> from ndmap.derivative import derivative
    >>> from ndmap.series import series
    >>> state = torch.tensor([0.0, 0.0])
    >>> knobs = torch.tensor([1.0, 1.0])
    >>> def fn(state, knobs):
    ...     x1, x2 = state
    ...     y1, y2 = knobs
    ...     return torch.stack([x1 + x2*y1, x2 + x1*y2])
    >>> def gn(state, knobs):
    ...     x1, x2 = state
    ...     y1, y2 = knobs
    ...     return torch.stack([y1 + y2*x1, y2 + y1*x2])
    >>> u = derivative((2, 2), lambda state, knobs: gn(fn(state, knobs), knobs), state, knobs)
    >>> u = series((2, 2), (2, 2), u)
    >>> v = identity((2, 2), [state, knobs], flag=True)
    >>> v = propagate((2, 2), (2, 2), v, [knobs], fn)
    >>> v = propagate((2, 2), (2, 2), v, [knobs], gn)
    >>> all(torch.allclose(x, y) for x, y in zip(u.values(), v.values()))
    True

    >>> import torch
    >>> state = torch.tensor([0.0, 0.0])
    >>> knobs = torch.tensor([1.0, 1.0])
    def fn(state, knobs):
    ...     x1, x2 = state
    ...     y1, y2 = knobs
    ...     return (x1**2 + x2**2)*(y1 + y2)
    >>> t = identity((2, 0), [state, knobs])
    >>> propagate((2, 2), (2, 0), t, [knobs], fn)
    [[tensor(0.)],
    [tensor([0., 0.])],
    [tensor([[4., 0.],
            [0., 4.]])]]

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    def auxiliary(*args) -> Tensor:
        state, *args = torch.stack([*args]).split(dimension)
        state = evaluate(data, [state, *args], epsilon=epsilon)
        if args:
            args = [arg + knob for arg, knob in zip(args, torch.cat(knobs))]
        return mapping(state, *args, *pars)

    value, *_ = data.values()
    delta = torch.cat([torch.zeros(i, dtype=value.dtype, device=value.device) for i in dimension])

    return series([*data.keys()], auxiliary, *delta, jacobian=jacobian)


@multimethod
def propagate(dimension:tuple[int, ...],
              order:tuple[int, ...],
              data:Table,
              knobs:Knobs,
              table:Table, *,
              intermediate:bool=True,
              jacobian:Optional[Callable]=None) -> Union[Table, Tensor]:
    """
    Propagate derivative table representation through a given table

    Note, can be used for composition of deririvative tables
    Composition should be performed for tables that map zero state to zero
    Also, tables are expected to have identical knobs

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        dimensions
    order: tuple[int, ...], non-negative
        maximum derivative orders
    data: Table
        input derivative table
    knobs: Knobs
        input parametric variables
    table: Table
        input table mapping approximation
    intermediate: bool, default=True
        flag to return intermediate derivatives
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Union[Table, Tensor]

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> from ndmap.evaluate import compare
    >>> def fn(x, l): q, p = x; return torch.stack([q + l*p, p])
    >>> x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> t = derivative(1, fn, x, 0.5)
    >>> compare(derivative(1, fn, x, 1.0), propagate((2, ), (1, ), t, [], t))
    True

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    def auxiliary(state, *knobs) -> Tensor:
        return evaluate(table, [state, *knobs])

    return propagate(dimension,
                     order,
                     data,
                     knobs,
                     auxiliary,
                     intermediate=intermediate,
                     jacobian=jacobian)
