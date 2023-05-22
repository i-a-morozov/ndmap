"""
Parametric fixed point
----------------------

Computation of dynamic and parametric fixed points

"""

from typing import TypeAlias
from typing import Callable
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from .derivative import derivative
from .signature import signature
from .signature import get
from .signature import set
from .propagate import identity
from .propagate import propagate


Mapping  : TypeAlias = Callable
Point    : TypeAlias = list[Tensor]
Delta    : TypeAlias = list[Tensor]
State    : TypeAlias = Tensor
Knobs    : TypeAlias = list[Tensor]
Table    : TypeAlias = list
Series   : TypeAlias = dict[tuple[int, ...], Tensor]
Signature: TypeAlias = Union[list[tuple[int, ...]], list[tuple[tuple[int, ...], float]]]


def newton(function:Mapping,
           guess:Tensor,
           *pars:tuple,
           solve:Callable=lambda jacobian, value: torch.linalg.pinv(jacobian) @ value,
           roots:Optional[Tensor]=None,
           jacobian:Callable=torch.func.jacfwd) -> Tensor:
    """
    Perform one Newton root search step

    Parameters
    ----------
    function: Mapping
        input function
    guess: Tensor
        initial guess
    *pars:
        additional function arguments
    solve: Callable, default=lambda jacobian, value: torch.linalg.pinv(jacobian) @ value
        linear solver
    roots: Optional[Tensor], default=None
        known roots to avoid
    jacobian: Callable, default=torch.func.jacfwd
        torch.func.jacfwd or torch.func.jacrev

    Returns
    -------
    Tensor

    """
    def auxiliary(x:Tensor, *xs) -> Tensor:
        return function(x, *xs)/(roots - x).prod(-1)

    value, jacobian = derivative(1,
                                 function if roots is None else auxiliary,
                                 guess,
                                 *pars,
                                 jacobian=jacobian)

    return guess - solve(jacobian, value)


def fixed_point(limit:int,
                function:Mapping,
                guess:Tensor,
                *pars:tuple,
                power:int=1,
                epsilon:Optional[float]=None,
                solve:Optional[Callable]=None,
                roots:Optional[Tensor]=None,
                jacobian:Callable=torch.func.jacfwd) -> Tensor:
    """
    Estimate (dynamical) fixed point

    Note, can be mapped over initial guess and/or other input function arguments if epsilon = None

    Parameters
    ----------
    limit: int, positive
        maximum number of newton iterations
    function: Mapping
        input mapping
    guess: Tensor
        initial guess
    *pars: tuple
        additional function arguments
    power: int, positive, default=1
        function power / fixed point order
    epsilon: Optional[float], default=None
        tolerance epsilon
    solve: Optional[Callable]
        linear solver(jacobian, vector)
    roots: Optional[Tensor], default=None
        known roots to avoid
    jacobian: Callable, default=torch.func.jacfwd
        torch.func.jacfwd or torch.func.jacrev

    Returns
    -------
    Tensor

    """
    if solve is None:
        def solve(jacobian, vector):
            return torch.linalg.pinv(jacobian) @ vector

    def auxiliary(state:Tensor) -> Tensor:
        local = torch.clone(state)
        for _ in range(power):
            local = function(local, *pars)
        return state - local

    point = torch.clone(guess)

    for _ in range(limit):
        point = newton(auxiliary, point, solve=solve, roots=roots, jacobian=jacobian)
        error = (point - guess).abs().max()
        guess = torch.clone(point)
        if epsilon is not None and error < epsilon:
            break

    return point


def check_point(power:int,
                function:Mapping,
                point:Tensor,
                *pars:tuple,
                epsilon:float=1.0E-12) -> bool:
    """
    Check fixed point candidate to have given prime period

    Parameters
    ----------
    power: int, positive
        function power / prime period
    function: Mapping
        input function
    point: Tensor
        fixed point candidate
    *pars:tuple
        additional function arguments
    epsilon: float, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    bool

    """
    def auxiliary(state:Tensor, power:int) -> Tensor:
        local = torch.clone(state)
        table = [local]
        for _ in range(power):
            local = function(local, *pars)
            table.append(local)
        return torch.stack(table)

    if power == 1:
        return True

    points = auxiliary(point, power)
    start, *points, end = points

    if (start - end).norm() > epsilon:
        return False

    return not torch.any((torch.stack(points) - point).norm(dim=-1) < epsilon)


def clean_point(power:int,
                function:Mapping,
                point:Tensor,
                *pars:tuple,
                epsilon:float=1.0E-12) -> bool:
    """
    Clean fixed point candidates

    Parameters
    ----------
    power: int, positive
        function power / prime period
    function: Mapping
        input function
    point: Tensor
        fixed point candidates
    *pars:tuple
        additional function arguments
    epsilon: float, optional, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    bool

    """
    point = point[torch.all(point.isnan().logical_not(), dim=1)]
    point = [x for x in point if check_point(power, function, x, *pars, epsilon=epsilon)]
    point = torch.stack(point)

    prime = []
    table = []

    for candidate in point:

        value = torch.linalg.eigvals(matrix(power, function, candidate, *pars))
        value = torch.stack(sorted(value, key=torch.norm))

        if not prime:
            prime.append(candidate)
            table.append(value)
            continue

        if all((torch.stack(prime) - candidate).norm(dim=-1) > epsilon):
            if all((torch.stack(table) - value).norm(dim=-1) > epsilon):
                prime.append(candidate)
                table.append(value)

    return torch.stack(prime)


def chain_point(power:int,
                function:Mapping,
                point:Tensor,
                *pars:tuple) -> Tensor:
    """
    Generate chain for a given fixed point.

    Note, can be mapped over point

    Parameters
    ----------
    power: int, positive
        function power
    function: Mapping
        input function
    point: Tensor
        fixed point
    *pars: tuple
        additional function arguments

    Returns
    -------
    Tensor

    """
    def auxiliary(state:Tensor) -> Tensor:
        local = torch.clone(state)
        table = [local]
        for _ in range(power - 1):
            local = function(local, *pars)
            table.append(local)
        return torch.stack(table)

    return auxiliary(point)


def matrix(power:int,
           function:Mapping,
           point:Tensor,
           *pars:tuple,
           jacobian:Callable=torch.func.jacfwd) -> Tensor:
    """
    Compute (monodromy) matrix around given fixed point.

    Parameters
    ----------
    power: int, positive
        function power / prime period
    function: Mapping
        input function
    point: Tensor
        fixed point candidate
    *pars: tuple
        additional function arguments
    jacobian: Callable, default=torch.func.jacfwd
        torch.func.jacfwd or torch.func.jacrev

    Returns
    -------
    Tensor

    """
    def auxiliary(state:Tensor) -> Tensor:
        local = torch.clone(state)
        for _ in range(power):
            local = function(local, *pars)
        return local

    return derivative(1, auxiliary, point, intermediate=False, jacobian=jacobian)


def parametric_fixed_point(order:tuple[int, ...],
                           state:State,
                           knobs:Knobs,
                           function:Mapping,
                           *pars:tuple,
                           power:int=1,
                           solve:Optional[Callable]=None,
                           jacobian:Callable=torch.func.jacfwd) -> Table:
    """
    Compute parametric fixed point.

    Parameters
    ----------
    order: tuple[int, ...], non-negative
        knobs derivative orders
    state: State
        state fixed point
    knobs: Knobs
        knobs value
    function:Callable
        input function
    *pars: tuple
        additional function arguments
    power: int, positive, default=1
        function power
    solve: Optional[Callable]
        linear solver(jacobian, )
    jacobian: Callable, default=torch.func.jacfwd
        torch.func.jacfwd or torch.func.jacrev

    Returns
    -------
    Table

    """
    if solve is None:
        def solve(jacobian, vector):
            return torch.linalg.pinv(jacobian) @ vector

    def auxiliary(*point) -> State:
        state, *knobs = point
        for _ in range(power):
            state = function(state, *knobs, *pars)
        return state

    def objective(value:Tensor, shape, index:tuple[int, ...]) -> Tensor:
        value = value.reshape(*shape)
        set(table, index, value)
        local = propagate(dimension,
                          index,
                          table,
                          knobs,
                          auxiliary,
                          intermediate=False,
                          jacobian=jacobian)
        return (value - local).flatten()

    dimension = (len(state), *(len(knob) for knob in knobs))
    order = (0, *order)

    table = identity(order, [state] + knobs, jacobian=jacobian)
    _, *array = signature(table)

    for index in array:
        guess = get(table, index)
        value = newton(objective,
                       guess.flatten(),
                       guess.shape,
                       index,
                       solve=solve,
                       jacobian=jacobian)
        value = value.reshape(*guess.shape)
        set(table, index, value.reshape(*guess.shape))

    return table
