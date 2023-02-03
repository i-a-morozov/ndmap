"""
Higher order partial derivatives computation with respect to one or several tensor-like variables.
Taylor series function approximation (derivative table and series function representation).
Parametric fixed point computation.

Ivan Morozov, 2022-2023

"""
from __future__ import annotations

import torch
import functorch

from multimethod import multimethod

from math import factorial
from math import prod

from torch import Tensor
from torch import Size

from typing import TypeAlias, Callable, Iterator, Optional


Mapping: TypeAlias = Callable
Point  : TypeAlias = list[Tensor]
Delta  : TypeAlias = list[Tensor]
State  : TypeAlias = Tensor
Knobs  : TypeAlias = list[Tensor]
Table  : TypeAlias = list
Series : TypeAlias = dict[tuple[int, ...], Tensor]


def multinomial(*sequence:int) -> float:
    """
    Compute multinomial coefficient for a given sequence (n, m, ...) of non-negative integers.

    (n + m + ...)! / (n! * m! * ... )

    Parameters
    ----------
    *sequence: int, non-negative
        input sequence of integers

    Returns
    ------
    multinomial coefficient (float)

    """
    return factorial(sum(sequence)) / prod(map(factorial, sequence))


def flatten(array:tuple, *, target:type=tuple) -> Iterator:
    """
    Flatten a nested tuple (or other selected target type container).

    Note, default target type is tuple

    Parameters
    ----------
    array: tuple
        input nested tuple
    target: type, optional, default=tuple
        target type to flatten

    Yields
    ------
    flattened tuple iterator (or selected target type container iterator)

    """
    if isinstance(array, target):
        for element in array:
            yield from flatten(element, target=target)
    else:
        yield array


@multimethod
def derivative(order:int,
               function:Callable,
               *args,
               intermediate:bool=True,
               jacobian:Callable=functorch.jacfwd) -> Table | Tensor:
    """
    Compute input function derivatives with respet to the first function argument upto a given order.

    Note, if intermediate flag is False, only the highest order derivative is returned

    Input function is expected to return a tensor or a (nested) list of tensors
    The first function argument is expected to be a tensor

    If the input function returns a tensor, output is called derivative table representation, [value, jacobian, hessian, ...]
    The first argument is called an evaluation point (State)

    Parameters
    ----------
    order: int, non-negative
        maximum derivative order
    function: Callable
        input function
    *args:
        input function arguments
    intermediate: bool, optional, default=True
        flag to return intermediate derivatives
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    -------
    function derivatives / derivative table representation (Table | Tensor)

    """
    def local(x, *xs):
        y = function(x, *xs)
        return (y, y) if intermediate else y

    for _ in range(order):
        def local(x, *xs, local=local):
            if not intermediate:
                return jacobian(local, has_aux=False)(x, *xs)
            y, ys = jacobian(local, has_aux=True)(x, *xs)
            return y, (ys, y)

    if not intermediate:
        return local(*args)

    y, ys = local(*args)

    return list(flatten(ys, target=tuple))


@multimethod
def derivative(order:tuple[int, ...],
               function:Callable,
               *args,
               intermediate:bool=True,
               jacobian:Callable=functorch.jacfwd) -> Table | Tensor:
    """
    Compute input function derivatives with respet to several first function arguments upto corresponding given orders.

    Note, if intermediate flag is False, only the highest (total) order derivative is returned

    Input function is expected to return a tensor or a (nested) list of tensors
    The first several function arguments are expected to be tensors

    If the input function returns a tensor, output is called derivative table representation
    List of several first arguments is called an evaluation point (Point)

    Parameters
    ----------
    order: tuple[int, ...], non-negative
        maximum derivative orders
    function: Callable
        input function
    *args:
        input function arguments
    intermediate: bool, optional, default=True
        flag to return intermediate derivatives
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    -------
    function derivatives / derivative table representation (Table | Tensor)

    """
    pars = [*args][len(order):]

    def fixed(*args):
        return function(*args, *pars)

    def build(order, value):
        def local(*args):
            return derivative(order, lambda x: fixed(*args, x), value, intermediate=intermediate, jacobian=jacobian)
        return local

    (order, value), *rest = zip(order, args)
    for degree, tensor in reversed(rest):
        def build(order, value, build=build(degree, tensor)):
            def local(*args):
                return derivative(order, lambda x: build(*args, x), value, intermediate=intermediate, jacobian=jacobian)
            return local

    return build(order, value)()


@multimethod
def derivative(order:int, function:Callable, point:Point, *pars, **kwargs) -> Table:
    """ (auxiliary) Compute input function derivatives at a given evaluation point upto a given order. """
    return derivative(order, function, *point, *pars, **kwargs)


@multimethod
def derivative(order:tuple[int, ...], function:Callable, point:Point, *pars, **kwargs) -> Table:
    """ (auxiliary) Compute input function derivatives at a given evaluation point upto corresponding given orders. """
    return derivative(order, function, *point, *pars, **kwargs)


@multimethod
def derivative(order:int, function:Callable, state:State, knobs:Knobs, *pars, **kwargs) -> Table:
    """ (auxiliary) Compute input function derivatives for given state and knobs upto a given order with respect to state. """
    return derivative(order, function, state, *knobs, *pars, **kwargs)


@multimethod
def derivative(order:tuple[int, ...], function:Callable, state:State, knobs:Knobs, *pars, **kwargs) -> Table:
    """ (auxiliary) Compute input function derivatives for given state and knobs upto corresponding given orders with respect to state and knobs. """
    return derivative(order, function, state, *knobs, *pars, **kwargs)


@multimethod
def signature(table:Table, *, factor:bool=False) -> list[tuple[int, ...]] | list[tuple[tuple[int, ...], float]]:
    """
    Compute derivative table bottom elements signatures.

    Note, signature elements corresponds to the bottom elements of a flattened input derivative table
    Bottom element signature is a tuple integers, derivative orders with respect to each tensor variable
    Optionaly return elements multiplication factors
    Given a signature (n, m, ...), corresponding multiplication factor is 1/n! * 1/m! * ...

    Parameters
    ----------
    table: Table
        input derivative table representation
    fator: bool, optional, default=True
        flag to return elements multipliation factors

    Returns
    ----------
    bottom table elements signatures (list[tuple[int, ...]] | list[tuple[tuple[int, ...], float]])
    """
    return [*flatten([signature([i], subtable, factor=factor) for i, subtable in enumerate(table)], target=list)]


@multimethod
def signature(index:list[int], table:Table, *, factor:bool=False):
    return [signature(index + [i], subtable, factor=factor) for i, subtable in enumerate(table)]


@multimethod
def signature(index:list[int], table:Tensor, *, factor:bool=False):
    value = 1.0
    for i, count in enumerate(index):
        value *= 1.0/factorial(count)
    return tuple(index) if not factor else (tuple(index), value)


def get(table:Table, index:tuple[int, ...]) -> Tensor | Table:
    """
    Get derivative table element at a given (bottom) element signature.

    Note, index can correspond to a bottom element or a subtable

    Parameters
    ----------
    table: Table
        input derivative table representation
    index: tuple[int, ...]
        element signature

    Returns
    ----------
    element value (Tensor | Table)

    """
    if isinstance(index, int):
        return table[index]

    *ns, n = index
    for i in ns:
        table = table[i]
    return table[n]


def set(table:Table, index:tuple[int, ...], value:Tensor | Table) -> None:
    """
    Set derivative table element at a given (bottom) element signature.

    Note, index can correspond to a bottom element or a subtable

    Parameters
    ----------
    table: Table
        input derivative table representation
    index: tuple[int, ...]
        element signature
    value: Tensor | Table
        element value

    Returns
    ----------
    None

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
    Apply function.

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
    ----------
    None

    """
    value = get(table, index)
    set(table, index, function(value))


@multimethod
def index(dimension:int,
          order:int,
          *,
          dtype:torch.dtype=torch.int64,
          device:torch.device=torch.device('cpu')) -> Tensor:
    """
    Generate monomial index table with repetitions for a given dimension and order (total monomial degree).

    Note, output length is dimension**degree

    Parameters
    ----------
    dimension: int, positive
        monomial dimension (number of variables)
    order: int, non-negative
        derivative order (total monomial degree)
    dtype: torch.dtype, optional, default=torch.int64
        data type
    device: torch.device, optional, default=torch.device('cpu')
        data device

    Returns
    ----------
    monomial index table with repetitions (Tensor)

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
          order:tuple[int, ...],
          *,
          dtype:torch.dtype=torch.int64,
          device:torch.device=torch.device('cpu')) -> Tensor:
    """
    Generate monomial index table with repetitions for given dimensions and corresponding orders (total monomial degrees).

    Note, output length is product(dimension**degree)

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        monomial dimensions
    order: tuple[int, ...], non-negative
        derivative orders (total monomial degrees)
    dtype: torch.dtype, optional, default=torch.int64
        data type
    device: torch.device, optional, default=torch.device('cpu')
        data device

    Returns
    ----------
    monomial index table with repetitions (Tensor)

    """
    def merge(total:tuple, *table:tuple) -> tuple:
        x, *xs = table
        return tuple(merge(total + i, *xs) for i in x) if xs else tuple(list(total + i) for i in x)

    x, *xs = [tuple(index(*pair).tolist()) for pair in zip(dimension + (0, ), order + (0, ))]

    return torch.tensor([*flatten(tuple(merge(i, *xs) for i in x))], dtype=dtype, device=device)


@multimethod
def series(dimension:tuple[int, ...],
           order:tuple[int, ...],
           table:Table) -> Series:
    """
    Generate series representation from a given derivative table representation.

    Note, table is expected to represent a vector valued function

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        dimensions
    order: tuple[int, ...], non-negative
        derivative orders
    table: Table
        derivative table representation

    Returns
    ----------
    series representation (Series)

    """
    series = {}

    for (count, factor), array in zip(signature(table, factor=True), flatten(table, target=list)):
        if not all(i <= j for i, j in zip(count, order)):
            continue
        count = index(dimension, count)
        array = factor*array.permute(*reversed(range(len(array.shape)))).flatten().reshape(-1, len(array))
        for key, value in zip(count, array):
            key = tuple(key.tolist())
            if key not in series:
                series[key]  = value
            else:
                series[key] += value

    return series


@multimethod
def series(index:tuple[int, ...],
           function:Callable,
           *args,
           jacobian:Callable=functorch.jacfwd) -> Series:
    """
    Generate series representation of a given input function upto a given monomial index.

    c(i, j, k, ...) * x**i * y**j * z**k * ... => {..., (i, j, k, ...) : c(i, j, k, ...), ...}

    Note, input function arguments are expected to be scalar tensors
    Function is expected to return a vector tensor

    Parameters
    ----------
    index: tuple[int, ...], non-negative
        list of monomial index, (i, j, k, ...)
    function: Callable,
        input function
    *args:
        input function arguments
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    -------
    series coeficients upto given index (Series)

    """
    return series((1, ) * len(args), index, derivative(index, function, *args, intermediate=True, jacobian=jacobian))


@multimethod
def series(index:list[tuple[int, ...]],
           function:Callable,
           *args,
           jacobian:Callable=functorch.jacfwd) -> Series:
    """
    Generate series representation of a given input function for a given set of monomial indices.

    c(i, j, k, ...) * x**i * y**j * z**k * ... => {..., (i, j, k, ...) : c(i, j, k, ...), ...}

    Note, input function arguments are expected to be scalar tensors
    Function is expected to return a vector tensor

    Parameters
    ----------
    index: list[tuple[int, ...]], non-negative
        list of monomial indices, [..., (i, j, k, ...), ...]
    function: Callable,
        input function
    *args:
        input function arguments
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    -------
    series coeficients for given indices (Series)

    """
    def factor(*index:int) -> float:
        return 1.0 / prod(map(factorial, index))

    return {i: factor(*i) * derivative(i, function, *args, intermediate=False, jacobian=jacobian) for i in index}


@multimethod
def series(index:list[tuple[int, ...]], function:Callable, point:Point, *pars, jacobian:Callable=functorch.jacfwd) -> Series:
    """ (auxiliary) Generate series representation of a given input function for a given set of monomial indices. """
    return series(index, function, *torch.cat(point), *pars, jacobian=jacobian)


@multimethod
def series(index:list[tuple[int, ...]], function:Callable, state:State, knobs:Knobs, *pars, jacobian:Callable=functorch.jacfwd) -> Series:
    """ (auxiliary) Generate series representation of a given input function for a given set of monomial indices. """
    return series(index, function, *state, *torch.cat(knobs), *pars, jacobian=jacobian)


def merge(probe:Series, other:Series) -> Series:
    """
    (series operation) Merge (sum) series.

    Parameters
    ----------
    probe, other: Series
        input series to merge

    Returns
    -------
    merged series (Series)

    """
    total = {key: value.clone() for key, value in probe.items()}
    for key, value in other.items():
        if key in total:
            total[key] += value
        else:
            total[key]  = value
    return total


def clean(probe:Series, *, epsilon:float=1.0E-16) -> Series:
    """
    (series operation) Clean series.

    Parameters
    ----------
    probe: Series
        input series to clean
    epsilon: float, optional, non-negative
        clean epsilon

    Returns
    -------
    cleaned series (Series)

    """
    return {key: value for key, value in probe.items() if torch.any(value > epsilon)}


def fetch(probe:Series, index:list[tuple[int, ...]]) -> Series:
    """
    (series operation) Fetch series.

    Parameters
    ----------
    probe: Series
        input series
    index: list[tuple[int, ...]], non-negative
        list of monomial indices, [..., (i, j, k, ...), ...]

    Returns
    -------
    fetched series (Series)

    """
    return {key: value for key, value in probe.items() if key in index}


def split(probe:Series) -> list[Series]:
    """
    (series operation) Split series.

    Note, coefficient values are assumed to be vector tensors.

    Parameters
    ----------
    probe: Series
        input series

    Returns
    -------
    splitted series (list[Series])

    """
    return [dict(zip(probe.keys(), value)) for value in torch.stack([*probe.values()]).T]


@multimethod
def evaluate(table:Table,
             delta:Delta) -> Tensor:
    """
    Evaluate input derivative table representation at a given delta deviation.

    Note, input table is expected to represent a vector or scalar valued function

    Parameters
    ----------
    table: Table
        input derivative table representation
    delta: Delta
        delta deviation

    Returns
    ----------
    value (Tensor)

    """
    return sum(evaluate([i], subtable, delta) for i, subtable in enumerate(table))


@multimethod
def evaluate(index:list[int], table:Table, delta:Delta):
    """ (auxiliary) Evaluate input derivative table representation at a given delta deviation. """
    return sum(evaluate(index + [i], subtable, delta) for i, subtable in enumerate(table))


@multimethod
def evaluate(index:list[int], table:Tensor, delta:Delta):
    """ (auxiliary) Evaluate input derivative table representation at a given delta deviation. """
    total = 1.0
    for count, order in enumerate(index):
        total *= 1.0/factorial(order)
        value  = delta[count]
        if value.ndim > 0:
            for _ in range(order): table @= value
        else:
            for _ in range(order): table *= value
    return total*table


@multimethod
def evaluate(series:Series,
             delta:Delta,
             epsilon:Optional[float]=None) -> Tensor:
    """
    Evaluate series representation at a given deviation delta.

    Note, input series is expected to represent a vector valued function
    For epsilon != None, fast evaluation is performed

    Parameters
    ----------
    series: Series
        input series representation
    delta: Delta
        delta deviation
    epsilon: float, optional, non-negative, default=None
        fast series evaluation

    Returns
    ----------
    value (Tensor)

    """
    state = torch.cat(delta)

    if epsilon is not None:
        state = epsilon + state
        index = torch.tensor([*series.keys()], dtype=torch.int64, device=state.device)
        value = torch.stack([*series.values()])
        return (value.T * (state**index).prod(-1)).sum(-1)

    total, *_ = series.values()
    total = torch.zeros_like(total)
    for key, value in series.items():
        local = torch.ones_like(state).prod()
        for i, x in zip(key, state):
            for _ in range(i): local = x * local
        total = total + value * local
    return total


def table(dimension:tuple[int, ...],
          order:tuple[int, ...],
          series:Series,
          *,
          epsilon:Optional[float]=None,
          jacobian:Callable=functorch.jacfwd) -> Table:
    """
    Generate derivative table representation from a given series representation.

    Note, table is generated by taking derivatives of evaluated series at zero deviation delta.
    For epsilon != None can be used for fast series evaluation, but can generate incorrect table

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        dimensions
    order: tuple[int, ...], non-negative
        maximum derivative orders
    series: Series
        input series representation
    epsilon: float, optional, non-negative, default=None
        fast series evaluation
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    ----------
    derivative table representation (Table)

    """
    def function(*args):
        return evaluate(series, [*args], epsilon=epsilon)

    value, *_ = series.values()
    delta = [torch.zeros(i, dtype=value.dtype, device=value.device) for i in dimension]

    return derivative(order, function, delta, intermediate=True, jacobian=jacobian)


def identity(order:tuple[int, ...],
             point:Point,
             *,
             flag:bool=False,
             jacobian:Callable=functorch.jacfwd) -> Table | Series:
    """
    Generate identity derivative table (or identity series).

    Note, identity table or series represent identity mapping

    Parameters
    ----------
    order: tuple[int, ...], non-negative
        maximum derivative orders
    point: Point
        evaluation point
    flag: bool, optional, default=False
        flag to return identity series instead of table
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    ----------
    identity derivative table or series (Table | Series)

    """
    table = derivative(order, lambda x, *xs: x, point, intermediate=True, jacobian=jacobian)

    if not flag:
        return table

    return series(tuple(map(len, point)), order, table)


@multimethod
def propagate(dimension:tuple[int, ...],
              order:tuple[int, ...],
              data:Table,
              knobs:Knobs,
              mapping:Mapping,
              *pars,
              intermediate:bool=True,
              jacobian:Callable=functorch.jacfwd) -> Table | Tensor:
    """
    Propagate derivative table representation through a given mapping.

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
    *pars:
        additional mapping fixed arguments
    intermediate: bool, optional, default=True
        flag to return intermediate derivatives
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    ----------
    propagated derivative table (Table | Tensor)

    """
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
              *pars,
              epsilon:Optional[float]=None,
              jacobian:Callable=functorch.jacfwd) -> Series:
    """
    Propagate series representation through a given mapping.

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
    *pars:
        additional mapping fixed arguments
    epsilon: float, optional, non-negative, default=None
        fast series evaluation
    intermediate: bool, optional, default=True
        flag to return intermediate derivatives
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    ----------
    propagated series (Series)

    """
    def auxiliary(*args) -> Tensor:
        state, *args = torch.stack([*args]).split(dimension)
        state = evaluate(data, [state, *args], epsilon=epsilon)
        args = [arg + knob for arg, knob in zip(args, torch.cat(knobs))]
        return mapping(state, *args, *pars)

    value, *_ = series.values()
    delta = torch.cat([torch.zeros(i, dtype=value.dtype, device=value.device) for i in dimension])

    return series([*data.keys()], auxiliary, *delta, jacobian=jacobian)


def newton(function:Mapping,
           guess:Tensor,
           *pars,
           solve:Callable=lambda jacobian, value: torch.linalg.pinv(jacobian) @ value,
           roots:Optional[Tensor]=None,
           jacobian:Callable=functorch.jacfwd) -> Tensor:
    """
    Perform one Newton root search step.

    Parameters
    ----------
    function: Mapping
        input function
    guess: Tensor
        initial guess
    *pars:
        additional function arguments
    solve: Callable, optional, default=lambda jacobian, value: torch.linalg.pinv(jacobian) @ value
        linear solver
    roots: Optional[Tensor], optional, default=None
        known roots to avoid
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    ----------
    updated guess (Tensor)

    """
    def auxiliary(x:Tensor, *xs) -> Tensor:
        return function(x, *xs)/(roots - x).prod(-1)

    value, jacobian = derivative(1, function if roots is None else auxiliary, guess, *pars, jacobian=jacobian)

    return guess - solve(jacobian, value)


def fixed_point(limit:int,
                function:Mapping,
                guess:Tensor,
                *pars,
                power:int=1,
                epsilon:Optional[float]=None,
                solve:Callable=lambda jacobian, value: torch.linalg.pinv(jacobian) @ value,
                roots:Optional[Tensor]=None,
                jacobian:Callable=functorch.jacfwd) -> Tensor:
    """
    Estimate (dynamical) fixed point.

    Note, can be mapped over initial guess and/or other input function arguments if epsilon = None

    Parameters
    ----------
    limit: int, positive
        maximum number of newton iterations
    function: Mapping
        input function
    guess: Tensor
        initial guess
    *pars:
        additional function arguments
    power: int, optional, positive, default=1
        function power
    epsilon: Optional[float], optional, default=None
        tolerance epsilon
    solve: Callable, optional, default=lambda jacobian, value: torch.linalg.pinv(jacobian) @ value
        linear solver
    roots: Optional[Tensor], optional, default=None
        known roots to avoid
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    ----------
    estimated fixed point (Tensor)

    """
    def auxiliary(state:Tensor) -> Tensor:
        local = torch.clone(state)
        for _ in range(power):
            local = function(local, *pars)
        return state - local

    point = torch.clone(guess)

    for count in range(limit):
        point = newton(auxiliary, point, solve=solve, roots=roots, jacobian=jacobian)
        error = (point - guess).abs().max()
        guess = torch.clone(point)
        if epsilon is not None and error < epsilon:
            break

    return point


def check_point(power:int,
                function:Mapping,
                point:Tensor,
                *pars,
                epsilon:float=1.0E-12) -> bool:
    """
    Check fixed point candidate to have given prime period.

    Parameters
    ----------
    power: int, positive
        function power / prime period
    function: Mapping
        input function
    point: Tensor
        fixed point candidate
    *pars:
        additional function arguments
    epsilon: float, optional, default=1.0E-12
        tolerance epsilon

    Returns
    ----------
    check result (bool)

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


def matrix(power:int,
           function:Mapping,
           point:Tensor,
           *pars,
           jacobian:Callable=functorch.jacfwd) -> Tensor:
    """
    Compute matrix around a given fixed point.

    Parameters
    ----------
    power: int, positive
        function power / prime period
    function: Mapping
        input function
    point: Tensor
        fixed point candidate
    *pars:
        additional function arguments
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    ----------
    matrix (Tensor)

    """
    def auxiliary(state:Tensor) -> Tensor:
        local = torch.clone(state)
        for _ in range(power):
            local = function(local, *pars)
        return local

    return derivative(1, auxiliary, point, intermediate=False, jacobian=jacobian)


def clean_point(power:int,
                function:Mapping,
                point:Tensor,
                *pars,
                epsilon:float=1.0E-12) -> bool:
    """
    Clean fixed point candidates.

    Parameters
    ----------
    power: int, positive
        function power / prime period
    function: Mapping
        input function
    point: Tensor
        fixed point candidates
    *pars:
        additional function arguments
    epsilon: float, optional, default=1.0E-12
        tolerance epsilon

    Returns
    ----------
    cleaned result (bool)

    """
    point = point[torch.all(point.isnan().logical_not(), dim=1)]
    point = torch.stack([candidate for candidate in point if check_point(power, function, candidate, *pars, epsilon=epsilon)])

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
                *pars) -> Tensor:
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
    *pars:
        additional function arguments

    Returns
    ----------
    chain (Tensor)

    """
    def auxiliary(state:Tensor) -> Tensor:
        local = torch.clone(state)
        table = [local]
        for _ in range(power - 1):
            local = function(local, *pars)
            table.append(local)
        return torch.stack(table)

    return auxiliary(point)


def parametric_fixed_point(order:tuple[int, ...],
                           state:State,
                           knobs:Knobs,
                           function:Mapping,
                           *pars,
                           power:int=1,
                           solve:Callable=lambda jacobian, value: torch.linalg.pinv(jacobian) @ value,
                           jacobian:Callable=functorch.jacfwd) -> Table:
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
    *pars:
        additional function arguments
    power: int, optional, positive, default=1
        function power
    solve: Callable, optional, default=lambda jacobian, value: torch.linalg.pinv(jacobian) @ value
        linear solver
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev

    Returns
    ----------
    parametric fixed point (Table)

    """
    def auxiliary(*point) -> State:
        state, *knobs = point
        for _ in range(power):
            state = function(state, *knobs, *pars)
        return state

    def objective(value:Tensor, shape:Size, index:tuple[int, ...]) -> Tensor:
        value = value.reshape(*shape)
        set(table, index, value)
        local = propagate(dimension, index, table, knobs, auxiliary, intermediate=False, jacobian=jacobian)
        return (value - local).flatten()

    dimension = (len(state), *(len(knob) for knob in knobs))
    order = (0, *order)

    table = identity(order, [state] + knobs, jacobian=jacobian)
    _, *array = signature(table)

    for index in array:
        guess = get(table, index)
        value = newton(objective, guess.flatten(), guess.shape, index, solve=solve, jacobian=jacobian).reshape(*guess.shape)
        set(table, index, value.reshape(*guess.shape))

    return table


class Jet():
    """
    Convenience class to work with jets (evaluation point & derivative table)

    Returns
    ----------
    Jet class instance.

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        dimensions
    order: tuple[int, ...], non-negative
        maximum derivative orders
    initialize : bool
        flag to initialize identity derivative table, optional, default=True
    point: Optional[Point]
        evaluation point, optional, default=None
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev
    dtype: torch.dtype, optional, default=torch.float64
        data type
    device: torch.device, optional, default=torch.device('cpu')
        data device

    Attributes
    ----------
    dimension: tuple[int, ...], positive
        dimensions
    order: tuple[int, ...], non-negative
        maximum derivative orders
    initialize : bool
        flag to initialize identity derivative table, optional, default=True
    point: Optional[Point]
        evaluation point, optional, default=None
    jacobian: Callable, optional, default=functorch.jacfwd
        functorch.jacfwd or functorch.jacrev
    dtype: torch.dtype, optional, default=torch.float64
        data type
    device: torch.device, optional, default=torch.device('cpu')
        data device
    state: State
        state
    knobs: Knobs
        knobs
    table: Table
        table representation
    series: Series
        (property) series representation
    signature: list[tuple[int, ...]]
        (property) derivative table elements bottom elements signatures
    parametetric: Table
        (property) parametric table


    Methods
    ----------
    __init__(self, dimension:tuple[int, ...], order:tuple[int, ...], *, initialize:bool=True, point:Optional[Point]=None, jacobian:Callable=functorch.jacfwd, dtype:torch.dtype=torch.float64, device:torch.device=torch.device('cpu')) -> None
        Jet initialization.
    evaluate(self, delta:Delta) -> Tensor
        Evaluate jet derivative table at a given delta deviation.
    signature(self) -> list[tuple[int, ...]]
        Compute derivative table elements bottom elements signatures.
    series(self) -> Series
        Series representation.
    parametetric(self) -> Table
        Get parametric table (first subtable).
    from_mapping(cls, dimension:tuple[int, ...], order:tuple[int, ...], point:Point, function:Mapping, *args, jacobian:Callable=functorch.jacfwd, dtype:torch.dtype=torch.float64, device:torch.device=torch.device('cpu')) -> Jet
        Jet initialization from mapping.
    from_table(cls, dimension:tuple[int, ...], order:tuple[int, ...], point:Point, table:Table, jacobian:Callable=functorch.jacfwd, dtype:torch.dtype=torch.float64, device:torch.device=torch.device('cpu')) -> Jet
        Jet initialization from table.
    from_series(cls, dimension:tuple[int, ...], order:tuple[int, ...], point:Point, series:Series, jacobian:Callable=functorch.jacfwd, dtype:torch.dtype=torch.float64, device:torch.device=torch.device('cpu')) -> Jet
        Jet initialization from series.
    propagate(self, function:Mapping, *pars) -> Jet
        Propagate jet.
    compliant(self, other:Jet) -> bool
        Check jets are compliant (can be composed).
    compose(self, other:Jet) -> Jet
        Compose jets (evaluate other jet at self jet).
    __bool__(self) -> bool
        Check if table is not None
    __eq__(self, other:Jet) -> bool:
        Compare jets.
    __getitem__(self, index:tuple[int, ...]) -> Tensor | Table
        Get item (derivative table bottom element or subtable).
    __setitem__(self, index:tuple[int, ...], value:Tensor | Table) -> None
        Set item (derivative table bottom element or subtable).
    __iter__(self)
        Jet iteration (use for unpacking).
    __len__(self) -> int
        Jet length (signature length / number of tensors).
    __call__(self, delta:Delta) -> Tensor
        Evaluate jet derivative table at a given delta deviation.
    __matmul__(self, other:Jet) -> Jet
        Compose jets (evaluate other jet at self jet).
    __repr__(self) -> str
        String representation.

    """
    def __init__(self,
                 dimension:tuple[int, ...],
                 order:tuple[int, ...],
                 *,
                 initialize:bool=True,
                 point:Optional[Point]=None,
                 jacobian:Callable=functorch.jacfwd,
                 dtype:torch.dtype=torch.float64,
                 device:torch.device=torch.device('cpu')) -> None:
        """
        Jet initialization.

        Parameters
        ----------
        dimension: tuple[int, ...], positive
            dimensions
        order: tuple[int, ...], non-negative
            maximum derivative orders
        initialize : bool
            flag to initialize identity derivative table, optional, default=True
        point: Optional[Point]
            evaluation point, optional, default=None
        jacobian: Callable, optional, default=functorch.jacfwd
            functorch.jacfwd or functorch.jacrev
        dtype: torch.dtype, optional, default=torch.float64
            data type
        device: torch.device, optional, default=torch.device('cpu')
            data device

        Returns
        ----------
        None

        """
        self.dimension:tuple[int, ...] = dimension
        self.order:tuple[int, ...] = order
        self.initialize:bool = initialize
        self.point:Point = point
        self.jacobian:Callable = jacobian
        self.dtype:torch.dtype = dtype
        self.device:torch.device = device

        if self.point is None:
            self.point = [torch.zeros(i, dtype=self.dtype, device=self.device) for i in self.dimension]

        state, *knobs = self.point
        self.state:State = state
        self.knobs:Knobs = knobs

        self.table:Table = None if self.initialize is None else identity(self.order, self.point, flag=False, jacobian=self.jacobian)


    def evaluate(self, delta:Delta) -> Tensor:
        """
        Evaluate jet derivative table at a given delta deviation.

        Parameters
        ----------
        delta: Delta
            delta deviation

        Returns
        ----------
        value (Tensor)

        """
        return evaluate(self.table, delta)


    @property
    def signature(self) -> list[tuple[int, ...]]:
        """
        Compute derivative table elements bottom elements signatures.

        Parameters
        ----------
        None

        Returns
        ----------
        bottom table elements signatures (list[tuple[int, ...]])

        """
        return signature(self.table, factor=False)


    @property
    def series(self) -> Series:
        """
        Series representation.

        Parameters
        ----------
        None

        Returns
        ----------
        series representation (Series)

        """
        return series(self.dimension, self.order, self.table)


    @property
    def parametetric(self) -> Table:
        """
        Get parametric table (first subtable).

        Parameters
        ----------
        None

        Returns
        ----------
        parametric table (Table)

        """
        table, *_ = self.table
        return table


    @parametetric.setter
    def parametetric(self, value:Table) -> None:
        """
        Set parametric table (first subtable).

        Parameters
        ----------
        value: Table
            parametric table

        Returns
        ----------
        None

        """
        set(self.table, (0, ), value)


    @classmethod
    def from_mapping(cls,
                     dimension:tuple[int, ...],
                     order:tuple[int, ...],
                     point:Point,
                     function:Mapping,
                     *args,
                     jacobian:Callable=functorch.jacfwd,
                     dtype:torch.dtype=torch.float64,
                     device:torch.device=torch.device('cpu')) -> Jet:
        """
        Jet initialization from mapping.

        Parameters
        ----------
        dimension: tuple[int, ...], positive
            dimensions
        order: tuple[int, ...], non-negative
            maximum derivative orders
        point: Point
            evaluation point
        function: Mapping
            input function
        *args:
            additional function arguments
        jacobian: Callable, optional, default=functorch.jacfwd
            functorch.jacfwd or functorch.jacrev
        dtype: torch.dtype, optional, default=torch.float64
            data type
        device: torch.device, optional, default=torch.device('cpu')
            data device

        Returns
        ----------
        jet (Jet)

        """
        jet = cls(dimension, order, initialize=False, point=point, jacobian=jacobian, dtype=dtype, device=device)
        jet.table:Table = derivative(order, function, *point, *args, intermediate=True, jacobian=jacobian)
        return jet


    @classmethod
    def from_table(cls,
                   dimension:tuple[int, ...],
                   order:tuple[int, ...],
                   point:Point,
                   table:Table,
                   jacobian:Callable=functorch.jacfwd,
                   dtype:torch.dtype=torch.float64,
                   device:torch.device=torch.device('cpu')) -> Jet:
        """
        Jet initialization from table.

        Parameters
        ----------
        dimension: tuple[int, ...], positive
            dimensions
        order: tuple[int, ...], non-negative
            maximum derivative orders
        point: Point
            evaluation point
        table: Table
            input (derivative) table
        jacobian: Callable, optional, default=functorch.jacfwd
            functorch.jacfwd or functorch.jacrev
        dtype: torch.dtype, optional, default=torch.float64
            data type
        device: torch.device, optional, default=torch.device('cpu')
            data device

        Returns
        ----------
        jet (Jet)

        """
        jet = cls(dimension, order, initialize=False, point=point, jacobian=jacobian, dtype=dtype, device=device)
        jet.table:Table = table
        return jet


    @classmethod
    def from_series(cls,
                    dimension:tuple[int, ...],
                    order:tuple[int, ...],
                    point:Point,
                    series:Series,
                    jacobian:Callable=functorch.jacfwd,
                    dtype:torch.dtype=torch.float64,
                    device:torch.device=torch.device('cpu')) -> Jet:
        """
        Jet initialization from series.

        Parameters
        ----------
        dimension: tuple[int, ...], positive
            dimensions
        order: tuple[int, ...], non-negative
            maximum derivative orders
        point: list[Tensor]
            evaluation point
        series: Series
            input series
        jacobian: Callable, optional, default=functorch.jacfwd
            functorch.jacfwd or functorch.jacrev
        dtype: torch.dtype, optional, default=torch.float64
            data type
        device: torch.device, optional, default=torch.device('cpu')
            data device

        Returns
        ----------
        jet (Jet)

        """
        jet = cls(dimension, order, initialize=False, point=point, jacobian=jacobian, dtype=dtype, device=device)
        jet.table:Table = table(dimension, order, series, jacobian=jacobian, dtype=dtype, device=device)
        return jet


    def propagate(self,
                  function:Mapping,
                  *pars) -> Jet:
        """
        Propagate jet.

        Parameters
        ----------
        function: Mapping
            input function
        knobs: Knobs
            input function knobs
        *pars:
            additional function arguments

        Returns
        ----------
        propagated jet (Jet)

        """
        table = propagate(self.dimension, self.order, self.table, self.knobs, function, *pars, intermediate=True, jacobian=self.jacobian)
        return self.from_table(self.dimension, self.order, self.point, table, jacobian=self.jacobian, dtype=self.dtype, device=self.device)


    def compliant(self, other:Jet) -> bool:
        """
        Check jets are compliant (can be composed).

        Parameters
        ----------
        other: Jet
            other jet

        Returns
        ----------
        check result (bool)

        """
        if not all(i == j for i, j in zip(self.dimension, other.dimension)):
            return False

        if not all(i == j for i, j in zip(self.order, other.order)):
            return False

        return True


    def compose(self, other:Jet) -> Jet:
        """
        Compose jets (evaluate other jet at self jet).

        Parameters
        ----------
        other: Jet
            other jet

        Returns
        ----------
        composed jet (Jet)

        """
        def auxiliary(*args) -> Tensor:
            return other.evaluate([*args])

        return self.propagate(auxiliary)


    def __bool__(self) -> bool:
        """
        Check if table is not None

        Parameters
        ----------
        None

        Returns
        ----------
        check result (bool)

        """
        return self.table is not None


    def __eq__(self, other:Jet) -> bool:
        """
        Compare jets.

        Parameters
        ----------
        other: Jet
            other jet

        Returns
        ----------
        comparison result (bool)

        """
        if not self.compliant(other):
            return False

        if all(torch.allclose(x, y) for x, y in zip(self.series.values(), other.series.values())):
            return True

        return False


    def __getitem__(self, index:tuple[int, ...]) -> Tensor | Table:
        """
        Get item (derivative table bottom element or subtable).

        Parameters
        ----------
        index: tuple[int, ...]
            index

        Returns
        ----------
        value at index (Tensor|Table)

        """
        return get(self.table, index)


    def __setitem__(self, index:tuple[int, ...], value:Tensor | Table) -> None:
        """
        Set item (derivative table bottom element or subtable).

        Parameters
        ----------
        index: tuple[int, ...]
            index
        value: Tensor | Table
            value to set

        Returns
        ----------
        None

        """
        set(self.table, index, value)


    def __iter__(self):
        """
        Jet iteration (use for unpacking).

        """
        return flatten(self.table, target=list)


    def __len__(self) -> int:
        """
        Jet length (signature length / number of tensors).

        Parameters
        ----------
        None

        Returns
        ----------
        length (int)

        """
        return len(self.signature)


    def __call__(self, delta:Delta) -> torch.Tensor:
        """
        Evaluate jet derivative table at a given delta deviation.

        Parameters
        ----------
        delta: Delta
            delta deviation

        Returns
        ----------
        value (Tensor)

        """
        return self.evaluate(delta)


    def __matmul__(self, other:Jet) -> Jet:
        """
        Compose jets (evaluate other jet at self jet).

        Parameters
        ----------
        other: Jet
            other jet

        Returns
        ----------
        composed jet (Jet)

        """
        return self.compose(other)


    def __repr__(self) -> str:
        """
        String representation.

        Parameters
        ----------
        None

        Returns
        ----------
        string representation (str)

        """
        return f'Jet({self.dimension}, {self.order})'

about = """
# ndtorch is a collection of tools for computation of higher order derivatives (including all intermediate derivatives) with applications in nonlinear dynamics (accelerator physics)

# Derivatives of f(x, y, ...) with respect to several tensor arguments x, y, ... and corresponding derivative orders n, m, ...
# Input function f(x, y, ...) is expected to return a tensor or a (nested) list of tensors

# Derivatives and Taylor approximation of f(x, y, ...) with respect to scalar and/or vector tensor arguments x, y, ... and corresponding derivative orders n, m, ...
# Evaluation of Taylor approximation at a given deviation point
# Input function f(x, y, ...) is expected to return a scalar or vector tensor

# Derivatives are computed by nesting functorch jacobian function
# Higher order derivatives with respect to several arguments can be computed using different orders for each argument
# Derivatives can be used as a surrogate model for a given function with a vector tensor output (Taylor series approximation and evaluation)
# Tools for fixed point computation are avaliable for mappings, including (parametric) derivatives
# Mapping is a function f(x, y, ...) that returns a tensor with the same shape as its first argument x (assumed to be a vector)
# The first argument is referred as state, other arguments used for computation of derivatives are collectively referred as knobs
# State and each knob are assumed to be vector-like tensors

# Given a function f(x) with a single tensor argument x that returns a tensor or a (nested) list of tensors f(x)
# Derivatives with respect to x can be computed upto a given order
# By default, all intermediate derivatives are returned, but it is possible to return only the highest order derivative

# Derivatives are computed by nesting functorch jacobian function (forward or reverse jacobians can be used)
# Nesting of jacobian generates redundant computations starting from the second order for functions with non-scalar arguments
# Identical partial derivatives are recomputed several times
# Hence, computation overhead is exponentially increasing for higher orders

# Note, there is no redundant computations in the case of the first order derivatives
# Redundancy can be avoided if function argument is treated as separate scalar tensors or vector tensors with one component

# If an input function returns a tensor, then its derivatives are f(x), Dx f(x), Dxx f(x), ... (value, jacobian, hessian, ...)
# Note, function value is treated as zeros order derivative
# If returned value is a scalar tensor, jacobian is a vector (gradient) and hessian is a matrix
# Given the derivatives, the input function Taylor approximation at evaluation point can be constructed
# f(x + dx) = 1/0! * f(x) + 1/1! * Dx f(x) @ dx + 1/2! * Dxx f(x) @ dx @ dx + ...
# A list t(f, x) = [f(x), Dx f(x), Dxx f(x), ...] is called a derivative table, x is an evaluation point, dx is a deviation from x
# Derivative order can be deduced for table structure, table length in the case of a single tensor argument
# Note, function to evaluate derivative table is avaliable for input functions with scalar or vector tensor output

# Table representation is redundant in general, instead the result can be represented by series, collection of monomial coefficients
# c(n1, n2, ...) * x1^n1 * x2^n2 * ... ~ (n1, n2, ...): c(n1, n2, ...), tuple (n1, n2, ...) is called a monomial index
# Series representaion can be computed from a given table representation, table can be also computed from given series
# Both table and series representations can be evaluated for a given deviation dx
# Both table or series can be used to approximate original function (surrogate model)
# Note, input function is assumed to return a vector tensor for series representation

# For a function with several tensor arguments f(x, y, z, ...), derivatives can be computed separately with respect to each argument
# Derivative orders and arguments shapes can be different
# Derivative table has a nested structure in the case with several tensor arguments
# Combined monomial x1^n1 * x2^n2 * ... * y1^m1 * y2^m2 * ... * z1^k1 * z2^k2 * ... is used in series representation
# Again, for series representation, input function is assumed to return a vector
# Note, the order of derivatives is Dx...xy...yz...z = Dx...x Dy...y Dz...z f
# Derivatives are computed starting from the last argument
# Derivative table structure for f(x), f(x, y) and f(x, y, z) is shown below
# Similar structure holds for the case with more arguments

# f(x)
# t(f, x)
# [f, Dx f, Dxx f, ...]

# f(x, y)
# t(f, x, y)
# [
#     [    f,     Dy f,     Dyy f, ...],
#     [ Dx f,  Dx Dy f,  Dx Dyy f, ...],
#     [Dxx f, Dxx Dy f, Dxx Dyy f, ...],
#     ...
# ]

# f(x, y, z)
# t(f, x, y, z)
# [
#     [
#         [         f,          Dz f,          Dzz f, ...],
#         [      Dy f,       Dy Dz f,       Dy Dzz f, ...],
#         [     Dyy f,      Dyy Dz f,      Dyy Dzz f, ...],
#         ...
#     ],
#     [
#         [      Dx f,       Dx Dz f,       Dx Dzz f, ...],
#         [   Dx Dy f,    Dx Dy Dz f,    Dx Dy Dzz f, ...],
#         [  Dx Dyy f,   Dx Dyy Dz f,   Dx Dyy Dzz f, ...],
#         ...
#     ],
#     [
#         [    Dxx f,     Dxx Dz f,     Dxx Dzz f, ...],
#         [ Dxx Dy f,  Dxx Dy Dz f,  Dxx Dy Dzz f, ...],
#         [Dxx Dyy f, Dxx Dyy Dz f, Dxx Dyy Dzz f, ...],
#         ...
#     ],
#     ...
# ]

# Each tensor at the bottom level is associated with a signature, e.g. Dxxyzz f = Dxx Dy Dzz f ~ (2, 1, 2)
# Signature is a tuple of corresponding orders
# Bottom table elements signatures can be computed with signature function
# Given a bottom element signature, it can be extracted or assigned a new value
# Note, subtables can also be extracted and modified

# Bottom table elements are related to corresponding monomials
# Given a signature (n, m, ...), possible monomial indices are (n1, n2, ..., m1, m2, ...) with n1 + n2 + ... = n, m1 + m2 + ... = m, ...
# Monomial index table with repetitions can be generated with index function
# Note, index table corresponds to reversed bottom table element if the number of function arguments is greater than one

# Mapping is a function f(x, y, z, ...) that returns a tensor with the same shape as its first argument x (assumed to be a vector)
# For mappings, x corresponds to dynamical variable (state), while other variables are parameters (knobs)
# Composition of mappings f and g is also a mapping h = g(f(x, y, z, ...), y, z, ...)
# Derivatives of h can be computed directly or by propagation of t(f, x, y, z, ...), derivatives of f(x, y, z, ...), thought g(x, y, z, ...)
# This can be done with propagate function which takes table (or series) as a parameter
# Dynamical variable x can be represented as identity table (or series), can be generated with identity function
# Note, for composition/propagation functions are expected to have identical signatures (state, *knobs, ...), fixed parameters can be different

# Given a mapping h = g(f(x, y, z, ...), y, z, ...), table t(h, x, y, z, ...) is equivalent to composition of tables t(f, x, y, z, ...) and t(g, x, y, z, ...)
# If mappings f and g have no constant terms
# This is a homomorphism property of truncated polynomials composition without constant terms

# Given a mapping f(x, y, z, ...), its (dynamical) fixed points of given period can be computed, x(n) := f^n(x(n), y, z, ...)
# Note, such fixed points might not exist for a general mapping
# Fixed points are computed with newton method as roots of F = x - f^n(x, y, z, ...) = 0

# Fixed points can depend of parameters/knobs (y, z, ...)
# Parametric fixed point can computed for given dynamical fixed point, i.e. derivatives of a fixed point with respect to parameters

# Jet class provides utilities for handling derivivative tables (generation, propagation, composition and other)
# Here jet is an evaluation point and corresponding derivative table (also variable dimensions, orders and othe parameters)

# Other functionality relevant to applications in nonlinear dynamics is planned to be added.

"""

def main():
    print(about)

if __name__ == '__main__':
    main()