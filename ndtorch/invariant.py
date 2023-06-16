"""
Invariant
---------

Direct computation of invariant(s)

"""

from typing import TypeAlias
from typing import Callable
from typing import Optional
from typing import Union

from multimethod import multimethod

import torch
from torch import Tensor

from .derivative import derivative
from .signature import signature
from .signature import set
from .signature import get
from .signature import chop
from .evaluate import evaluate
from .pfp import newton
from .pfp import propagate


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
def invariant(order:tuple[int, ...],
              state:State,
              knobs:Knobs,
              observable:Observable,
              mapping:Mapping, *,
              limit:int=10,
              threshold:float=1.0E-9,
              solve:Optional[Callable]=None,
              jacobian:Optional[Callable]=None) -> tuple[Table, list]:
    """
    Compute Taylor invariant for a given mapping

    Parameters
    ----------
    order: tuple[int, ...]
        computation order
    state: State
        state fixed point
    knobs: Knobs
        knobs value
    observable: Observable
        invariant guess
    mapping: Mapping
        mapping
    limit: int, positive, default=1
        number of extra passes limit
    threshold: float, default=1.0E-9
        threshold value
    solve: Optional[Callable]
        linear solver(matrix, vecor)
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    tuple[Table, list]

    Examples
    --------
    >>> import torch
    >>> from ndtorch.util import nest
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.series import series
    >>> from ndtorch.series import clean
    >>> from ndtorch.yoshida import yoshida
    >>> def fn(x, t): q, p = x ; return torch.stack([q, p - t*q - t*q**2])
    >>> def gn(x, t): q, p = x ; return torch.stack([q + t*p, p])
    >>> l = torch.tensor(1.0, dtype=torch.float64)
    >>> x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> t = derivative(2, nest(100, lambda x: yoshida(0, 1, True, [fn, gn])(x, l/100)), x)
    >>> f = lambda x: evaluate(t, [x])
    >>> i, _ = invariant((3, ), x, [], lambda x: 1/2*(x**2).sum(), f, threshold=1.0E-6)
    >>> clean(series((2, ), (3, ), i))
    {(2, 0): tensor([0.5000], dtype=torch.float64),
    (0, 2): tensor([0.5000], dtype=torch.float64),
    (3, 0): tensor([0.3333], dtype=torch.float64)}

    >>> from ndtorch.util import nest
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.series import series
    >>> from ndtorch.series import clean
    >>> from ndtorch.yoshida import yoshida
    >>> def fn(x, t, k): q, p = x ; k, = k ; return torch.stack([q, p - t*q - t*(1 + k)*q**2])
    >>> def gn(x, t, k): q, p = x ; k, = k ; return torch.stack([q + t*p, p])
    >>> l = torch.tensor(1.0, dtype=torch.float64)
    >>> x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> k = torch.tensor([0.0], dtype=torch.float64)
    >>> y = nest(100, lambda x, k: yoshida(0, 2, True, [fn, gn])(x, l/100, k))
    >>> t = derivative((2, 1), y, x, k)
    >>> f = lambda x, k: evaluate(t, [x, k])
    >>> i, _ = invariant((3, 1), x, [k], lambda x, k: 1/2*(x**2).sum(), f, threshold=1.0E-6)
    >>> clean(series((2, 1), (3, 1), i))
    {(2, 0, 0): tensor([0.5000], dtype=torch.float64),
    (0, 2, 0): tensor([0.5000], dtype=torch.float64),
    (3, 0, 0): tensor([0.3333], dtype=torch.float64),
    (3, 0, 1): tensor([0.3333], dtype=torch.float64)}

    Note
    ----
    Initial guess is required to avoid trivial solution

    """
    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.lstsq(matrix, vector.unsqueeze(1)).solution.squeeze()

    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    point = [state, *knobs]

    table = derivative(order, observable, state, *knobs)
    chop(table, threshold=threshold)

    def auxiliary(*point) -> State:
        state, *knobs = point
        state = mapping(state, *knobs)
        return evaluate(table, [state, *knobs])

    def objective(value:Tensor, shape, index:tuple[int, ...]) -> Tensor:
        value = value.reshape(*shape)
        set(table, index, value)
        local = derivative(index,
                           auxiliary,
                           *point,
                           intermediate=False,
                           jacobian=jacobian)
        return (value - local).flatten()

    _, *array = signature(table)

    for _ in range(limit):

        for i in array:
            guess = get(table, i)
            value = newton(objective,
                           guess.flatten(),
                           guess.shape,
                           i,
                           solve=solve,
                           jacobian=jacobian)
            value = value.reshape(*guess.shape)
            set(table, i, value.reshape(*guess.shape))

        final = derivative(order,
                           auxiliary,
                           *point,
                           intermediate=True,
                           jacobian=jacobian)

        array = [i for i in array if (get(table, i) - get(final, i)).abs().max() > threshold]
        if not array:
            break

    chop(table, threshold=threshold)

    return table, array


@multimethod
def invariant(order:tuple[int, ...],
              state:State,
              knobs:Knobs,
              observable:Observable,
              data:Table, *,
              limit:int=10,
              threshold:float=1.0E-9,
              solve:Optional[Callable]=None,
              jacobian:Optional[Callable]=None) -> tuple[Table, list]:
    """
    Compute Taylor invariant for a given derivative table

    Parameters
    ----------
    order: tuple[int, ...]
        computation order
    state: State
        state fixed point
    knobs: Knobs
        knobs value
    observable: Observable
        invariant guess
    data: Table
        table mapping representation
    limit: int, positive, default=1
        number of extra passes limit
    threshold: float, default=1.0E-9
        threshold value
    solve: Optional[Callable]
        linear solver(matrix, vecor)
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    tuple[Table, list]

    Examples
    --------
    >>> import torch
    >>> from ndtorch.util import nest
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.series import series
    >>> from ndtorch.series import clean
    >>> from ndtorch.yoshida import yoshida
    >>> def fn(x, t): q, p = x ; return torch.stack([q, p - t*q - t*q**2])
    >>> def gn(x, t): q, p = x ; return torch.stack([q + t*p, p])
    >>> l = torch.tensor(1.0, dtype=torch.float64)
    >>> x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> t = derivative(2, nest(100, lambda x: yoshida(0, 1, True, [fn, gn])(x, l/100)), x)
    >>> i, _ = invariant((3, ), x, [], lambda x: 1/2*(x**2).sum(), t, threshold=1.0E-6)
    >>> clean(series((2, ), (3, ), i))
    {(2, 0): tensor([0.5000], dtype=torch.float64),
    (0, 2): tensor([0.5000], dtype=torch.float64),
    (3, 0): tensor([0.3333], dtype=torch.float64)}

    >>> from ndtorch.util import nest
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.series import series
    >>> from ndtorch.series import clean
    >>> from ndtorch.yoshida import yoshida
    >>> def fn(x, t, k): q, p = x ; k, = k ; return torch.stack([q, p - t*q - t*(1 + k)*q**2])
    >>> def gn(x, t, k): q, p = x ; k, = k ; return torch.stack([q + t*p, p])
    >>> l = torch.tensor(1.0, dtype=torch.float64)
    >>> x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> k = torch.tensor([0.0], dtype=torch.float64)
    >>> y = nest(100, lambda x, k: yoshida(0, 2, True, [fn, gn])(x, l/100, k))
    >>> t = derivative((2, 1), y, x, k)
    >>> i, _ = invariant((3, 1), x, [k], lambda x, k: 1/2*(x**2).sum(), t, threshold=1.0E-6)
    >>> clean(series((2, 1), (3, 1), i))
    {(2, 0, 0): tensor([0.5000], dtype=torch.float64),
    (0, 2, 0): tensor([0.5000], dtype=torch.float64),
    (3, 0, 0): tensor([0.3333], dtype=torch.float64),
    (3, 0, 1): tensor([0.2808], dtype=torch.float64),
    (2, 1, 1): tensor([-0.2456], dtype=torch.float64),
    (1, 2, 1): tensor([-0.3826], dtype=torch.float64),
    (0, 3, 1): tensor([-0.1986], dtype=torch.float64)}

    Note
    ----
    Initial guess is required to avoid trivial solution

    """
    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.lstsq(matrix, vector.unsqueeze(1)).solution.squeeze()

    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    table = derivative(order, observable, state, *knobs)
    chop(table, threshold=threshold)

    def objective(value:Tensor, shape, index:tuple[int, ...]) -> Tensor:
        value = value.reshape(*shape)
        set(table, index, value)
        local = propagate(dimension,
                          index,
                          data,
                          [],
                          lambda state: evaluate(table, [state, *knobs]),
                          intermediate=False,
                          jacobian=jacobian)
        return (value - local).flatten()

    dimension = (len(state), *(len(knob) for knob in knobs))

    _, *array = signature(table)

    for _ in range(limit):

        for i in array:
            guess = get(table, i)
            value = newton(objective,
                           guess.flatten(),
                           guess.shape,
                           i,
                           solve=solve,
                           jacobian=jacobian)
            value = value.reshape(*guess.shape)
            set(table, i, value.reshape(*guess.shape))

        final = propagate(dimension,
                        order,
                        data,
                        [],
                        lambda state: evaluate(table, [state, *knobs]),
                        intermediate=True,
                        jacobian=jacobian)

        array = [i for i in array if (get(table, i) - get(final, i)).abs().max() > threshold]
        if not array:
            break

    chop(table, threshold=threshold)

    return table, array
