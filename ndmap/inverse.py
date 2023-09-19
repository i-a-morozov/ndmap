"""
Inverse
-------

Compute derivative table inverse

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


def inverse(order:tuple[int, ...],
            state:State,
            knobs:Knobs,
            data:Table, *,
            solve:Optional[Callable]=None,
            jacobian:Optional[Callable]=None) -> Table:
    """
    Compute inverse of input derivative table

    Note, input table is assumed to represent a mapping
    Which is assumed to map (parametric) zero to (parametric) zero
    Input state and knobs are deviations and should equal to zero

    Parameters
    ----------
    order: tuple[int, ...]
        computation order
    state: State
        state fixed point
    knobs: Knobs
        knobs value
    solve: Optional[Callable]
        linear solver(matrix, vecor)
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Table

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> from ndmap.propagate import propagate
    >>> def fn(x):
    ...     q, p = x
    ...     return torch.stack([q, p + q + q**2])
    >>> x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> t = derivative((2, ), fn, x)
    >>> inverse((2, ), x, [], t)
    [[],
    tensor([[ 1.,  0.],
            [-1.,  1.]], dtype=torch.float64),
    tensor([[[ 0.,  0.],
            [ 0.,  0.]],
    
            [[-2.,  0.],
            [ 0.,  0.]]], dtype=torch.float64)]

    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> from ndmap.propagate import propagate
    >>> def fn(x, k):
    ...     q, p = x
    ...     a, b = k
    ...     return torch.stack([q, p + (1 + a)*q + (1 + b)*q**2])
    >>> x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> k = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> t = derivative((2, 1), fn, x, k)
    >>> inverse((2, 1), x, [k], t)
    [[[], []],
    [tensor([[ 1.,  0.],
            [-1.,  1.]], dtype=torch.float64),
    tensor([[[ 0.,  0.],
            [ 0.,  0.]],
    
            [[-1.,  0.],
            [ 0.,  0.]]], dtype=torch.float64)],
    [tensor([[[ 0.,  0.],
            [ 0.,  0.]],
    
            [[-2.,  0.],
            [ 0.,  0.]]], dtype=torch.float64),
    tensor([[[[ 0.,  0.],
                [ 0.,  0.]],
    
            [[ 0.,  0.],
                [ 0.,  0.]]],
    
    
            [[[ 0.,  0.],
                [ 0.,  0.]],
    
            [[-2.,  0.],
                [ 0.,  0.]]]], dtype=torch.float64)]]

    """
    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.lstsq(matrix, vector.unsqueeze(1)).solution.squeeze()

    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    dimension = (len(state), *(len(knob) for knob in knobs))
    length, *_ = dimension

    def auxiliary(*point) -> State:
        state, *_ = point
        return state

    result = derivative(order, auxiliary, state, *knobs, jacobian=jacobian)

    def objective(values, index, sequence, shape, unique):
        values = values.reshape(-1, length)
        for key, value in zip(unique, values):
            unique[key] = value
        value = build(sequence, shape, unique)
        set(result, index, value)
        local = propagate(dimension,
                        index,
                        result,
                        knobs,
                        data,
                        intermediate=False,
                        jacobian=jacobian)
        if sum(index) == 1:
            local = local - torch.diag_embed(torch.ones_like(first(values)))
        *_, local = reduce(dimension, index, local)
        return torch.stack([*local.values()]).flatten()

    array = signature(data)

    for i in array:
        if not first(i):
            set(result, i, [])
            continue
        guess = get(result, i)
        sequence, shape, unique = reduce(dimension, i, guess)
        guess = torch.stack([*unique.values()]).flatten()
        values = newton(objective,
                        guess,
                        i,
                        sequence,
                        shape,
                        unique,
                        solve=solve,
                        jacobian=jacobian)
        for key, value in zip(unique, values.reshape(-1, length)):
            unique[key] = value
        set(result, i, build(sequence, shape, unique))

    return result
