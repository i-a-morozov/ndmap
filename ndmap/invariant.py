"""
Invariant
---------

Direct computation of invariant(s)

"""

from typing import TypeAlias
from typing import Callable
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from ndmap.util import first
from ndmap.derivative import derivative
from ndmap.signature import signature
from ndmap.signature import set
from ndmap.signature import get
from ndmap.signature import chop
from ndmap.index import reduce
from ndmap.index import build
from ndmap.pfp import newton
from ndmap.pfp import propagate


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


def invariant(order:tuple[int, ...],
              state:State,
              knobs:Knobs,
              observable:Observable,
              data:Table, *,
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
    >>> from ndmap.util import nest
    >>> from ndmap.derivative import derivative
    >>> from ndmap.series import series
    >>> from ndmap.series import clean
    >>> from ndmap.yoshida import yoshida
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

    >>> from ndmap.util import nest
    >>> from ndmap.derivative import derivative
    >>> from ndmap.series import series
    >>> from ndmap.series import clean
    >>> from ndmap.yoshida import yoshida
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
    (3, 0, 1): tensor([0.3333], dtype=torch.float64)}

    Note
    ----
    Input table is assumed to be origin preserving
    Initial guess is required to avoid trivial solution

    """
    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.lstsq(matrix, vector.unsqueeze(1)).solution.squeeze()

    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    table = derivative(order, observable, state, *knobs)
    chop(table, threshold=threshold)

    def objective(values, index, sequence, shape, unique):
        for key, value in zip(unique, values):
            unique[key] = value
        value = build(sequence, shape, unique)
        set(table, index, value)
        local = propagate(dimension,
                          index,
                          data,
                          knobs,
                          table,
                          intermediate=False,
                          jacobian=jacobian)
        *_, local = reduce(dimension, index, local)
        return values - torch.stack([*local.values()])

    dimension = (len(state), *(len(knob) for knob in knobs))

    start = 2
    array = signature(table)

    for i in array:
        if first(i) < start:
            set(table, i, [])
            continue
        guess = get(table, i)
        sequence, shape, unique = reduce(dimension, i, guess)
        guess = torch.stack([*unique.values()])
        values = newton(objective,
                        guess,
                        i,
                        sequence,
                        shape,
                        unique,
                        solve=solve,
                        jacobian=jacobian)
        for key, value in zip(unique, values):
            unique[key] = value
        set(table, i, build(sequence, shape, unique))

    final = propagate(dimension,
                      order,
                      data,
                      knobs,
                      table,
                      intermediate=True,
                      jacobian=jacobian)

    array = [i for i in array if first(i) > start and (get(table, i) - get(final, i)).abs().max() > threshold]

    chop(table, threshold=threshold)

    return table, array
