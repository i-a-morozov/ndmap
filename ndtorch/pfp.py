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
           solve:Optional[Callable]=None,
           roots:Optional[Tensor]=None,
           jacobian:Optional[Callable]=None) -> Tensor:
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
    solve: Optional[Callable]
        linear solver(matrix, vector)
    roots: Optional[Tensor], default=None
        known roots to avoid
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Tensor

    Examples
    --------
    >>> import torch
    >>> def fn(x):
    ...    return (x - 5)**2
    >>> x = torch.tensor(4.0)
    >>> for _ in range(16):
    ...     x = newton(fn, x, solve=lambda matrix, vector: vector/matrix)
    >>> torch.allclose(x, torch.tensor(5.0))
    True
    >>> def fn(x):
    ...    x1, x2 = x
    ...    return torch.stack([(x1 - 5)**2, (x2 + 5)**2])
    >>> x = torch.tensor([4.0, -4.0])
    >>> for _ in range(16):
    ...    x = newton(fn, x, solve=lambda matrix, vector: torch.linalg.pinv(matrix) @ vector)
    >>> torch.allclose(x, torch.tensor([5.0, -5.0]))
    True

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.pinv(matrix) @ vector

    def auxiliary(x:Tensor, *xs) -> Tensor:
        return function(x, *xs)/(roots - x).prod(-1)

    vector, matrix = derivative(1,
                                function if roots is None else auxiliary,
                                guess,
                                *pars,
                                jacobian=jacobian)

    return guess - solve(matrix, vector)


def fixed_point(limit:int,
                function:Mapping,
                guess:Tensor,
                *pars:tuple,
                power:int=1,
                epsilon:Optional[float]=None,
                solve:Optional[Callable]=None,
                roots:Optional[Tensor]=None,
                jacobian:Optional[Callable]=None) -> Tensor:
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
        linear solver(matrix, vector)
    roots: Optional[Tensor], default=None
        known roots to avoid
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Tensor

    Examples
    --------
    >>> from math import pi
    >>> import torch
    >>> from ndtorch.util import nest
    >>> mu = 2.0*pi*torch.tensor(1/3 - 0.01)
    >>> kq, ks, ko = torch.tensor([0.0, 0.25, -0.25])
    >>> def mapping(x):
    ...     q, p = x
    ...     q, p = q*mu.cos() + p*mu.sin(), p*mu.cos() - q*mu.sin()
    ...     q, p = q, p + (kq*q + ks*q**2 + ko*q**3)
    ...     return torch.stack([q, p])
    >>> xi = torch.tensor([1.25, 0.00])
    >>> fp = fixed_point(32, mapping, xi, power=3)
    >>> torch.allclose(fp, nest(3, mapping)(fp))
    True

    """
    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.pinv(matrix) @ vector

    jacobian = torch.func.jacfwd if jacobian is None else jacobian

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

    Examples
    --------
    >>> from math import pi
    >>> import torch
    >>> from ndtorch.util import nest
    >>> mu = 2.0*pi*torch.tensor(1/3 - 0.01)
    >>> kq, ks, ko = torch.tensor([0.0, 0.25, -0.25])
    >>> def mapping(x):
    ...     q, p = x
    ...     q, p = q*mu.cos() + p*mu.sin(), p*mu.cos() - q*mu.sin()
    ...     q, p = q, p + (kq*q + ks*q**2 + ko*q**3)
    ...     return torch.stack([q, p])
    >>> xi = torch.tensor([0.00, 0.00])
    >>> fp = fixed_point(32, mapping, xi, power=3)
    >>> torch.allclose(fp, nest(3, mapping)(fp))
    True
    >>> check_point(3, mapping, fp, epsilon=1.0E-6)
    False
    >>> xi = torch.tensor([1.25, 0.00])
    >>> fp = fixed_point(32, mapping, xi, power=3)
    >>> torch.allclose(fp, nest(3, mapping)(fp))
    True
    >>> check_point(3, mapping, fp, epsilon=1.0E-6)
    True

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

    Examples
    --------
    >>> from math import pi
    >>> import torch
    >>> from ndtorch.util import nest
    >>> mu = 2.0*pi*torch.tensor(1/3 - 0.01)
    >>> kq, ks, ko = torch.tensor([0.0, 0.25, -0.25])
    >>> def mapping(x):
    ...     q, p = x
    ...     q, p = q*mu.cos() + p*mu.sin(), p*mu.cos() - q*mu.sin()
    ...     q, p = q, p + (kq*q + ks*q**2 + ko*q**3)
    ...     return torch.stack([q, p])
    >>> xi = 2.0*torch.rand((128, 2), dtype=torch.float64) - 1.0
    >>> fp = torch.func.vmap(lambda x: fixed_point(32, mapping, x, power=3))(xi)
    >>> fp.shape
    torch.Size([128, 2])
    >>> clean_point(3, mapping, fp, epsilon=1.0E-12).shape
    torch.Size([2, 2])

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

    Examples
    --------
    >>> from math import pi
    >>> import torch
    >>> from ndtorch.util import nest
    >>> mu = 2.0*pi*torch.tensor(1/3 - 0.01)
    >>> kq, ks, ko = torch.tensor([0.0, 0.25, -0.25])
    >>> def mapping(x):
    ...     q, p = x
    ...     q, p = q*mu.cos() + p*mu.sin(), p*mu.cos() - q*mu.sin()
    ...     q, p = q, p + (kq*q + ks*q**2 + ko*q**3)
    ...     return torch.stack([q, p])
    >>> xi = 2.0*torch.rand((128, 2), dtype=torch.float64) - 1.0
    >>> fp = torch.func.vmap(lambda x: fixed_point(32, mapping, x, power=3))(xi)
    >>> fp = clean_point(3, mapping, fp, epsilon=1.0E-12)
    >>> torch.func.vmap(lambda x: chain_point(3, mapping, x))(fp).shape
    torch.Size([2, 3, 2])

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

    Examples
    --------
    >>> from math import pi
    >>> import torch
    >>> from ndtorch.util import nest
    >>> mu = 2.0*pi*torch.tensor(1/3 - 0.01)
    >>> kq, ks, ko = torch.tensor([0.0, 0.25, -0.25])
    >>> def mapping(x):
    ...     q, p = x
    ...     q, p = q*mu.cos() + p*mu.sin(), p*mu.cos() - q*mu.sin()
    ...     q, p = q, p + (kq*q + ks*q**2 + ko*q**3)
    >>>     return torch.stack([q, p])
    >>> xi = torch.tensor([1.25, 0.00])
    >>> fp = fixed_point(32, mapping, xi, power=3)
    >>> matrix(3, mapping, fp)
    tensor([[ 0.9770,  0.8199],
            [-0.5659,  0.5486]])

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
                           jacobian:Optional[Callable]=None) -> Table:
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
        linear solver(matrix, vecor)
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Table

    Examples
    --------
    >>> from math import pi
    >>> import torch
    >>> from ndtorch.util import flatten
    >>> from ndtorch.util import nest
    >>> from ndtorch.evaluate import evaluate
    >>> from ndtorch.propagate import propagate
    >>> mu = 2.0*pi*torch.tensor(1/5 - 0.01, dtype=torch.float64)
    >>> k = torch.tensor([0.25, -0.25], dtype=torch.float64)
    >>> def mapping(x, k):
    ...     q, p = x
    ...     a, b = k
    ...     q, p = q*mu.cos() + p*mu.sin(), p*mu.cos() - q*mu.sin()
    ...     return torch.stack([q, p + a*q**2 + b*q**3])
    >>> xi = torch.tensor([0.75, 0.25], dtype=torch.float64)
    >>> fp = fixed_point(32, mapping, xi, k, power=5)
    >>> fp
    tensor([0.7279, 0.4947], dtype=torch.float64)
    >>> torch.allclose(fp, nest(5, lambda x: mapping(x, k))(fp))
    True
    >>> torch.linalg.eigvals(matrix(5, mapping, fp, k))
    tensor([1.3161+0.j, 0.7598+0.j], dtype=torch.complex128)
    >>> pfp = parametric_fixed_point((4, ), fp, [k], mapping, power=5)
    >>> out = propagate((2, 2), (0, 4), pfp, [k], lambda x, k: nest(5, mapping, k)(x, k))
    >>> all(torch.allclose(x, y) for x, y in zip(*map(lambda x: flatten(x, target=list), (pfp, out))))
    True
    >>> dk = torch.tensor([0.01, -0.01], dtype=torch.float64)
    >>> fp = fixed_point(32, mapping, xi, k + dk, power=5)
    >>> fp
    tensor([0.7163, 0.4868], dtype=torch.float64)
    >>> evaluate(pfp, [fp, dk])
    tensor([0.7163, 0.4868], dtype=torch.float64)

    """
    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.pinv(matrix) @ vector

    jacobian = torch.func.jacfwd if jacobian is None else jacobian

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
