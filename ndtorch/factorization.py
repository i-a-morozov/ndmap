"""
Factorization
-------------

Factorization related utilities

"""

from typing import TypeAlias
from typing import Callable
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from .util import first
from .derivative import derivative
from .signature import signature
from .signature import get
from .signature import set
from .index import reduce
from .index import build
from .evaluate import evaluate
from .taylor import taylor


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


def hamiltonian(order:tuple[int, ...],
                state:State,
                knobs:Knobs,
                table:Table, *,
                start:Optional[int]=None,
                count:Optional[int]=None,
                solve:Optional[Callable]=None,
                jacobian:Optional[Callable]=None) -> Table:
    """
    Compute hamiltonian taylor representation of a given near identity table

    Note, taylor integrator is used to construct hamiltonian
    Number of terms in the expansion is inferred from the input order
    The input order is assumed to correspond to the input table
    The input table represents a near identity mapping
    Thus, the first term of the hamiltonian is a degree three polynomial
    Starting order can be passed on input, e.g. when preceding orders are identically zero

    Parameters
    ----------
    order: tuple[int, ...]
        table order
    state: State
        state
    knobs: Knobs
        knobs
    table: Table
        near identity mapping taylor representation
    start: Optional[int]
        starting order (degree)
    count: Optional[int]
        number of terms to use in expansion (derived from input order if None)
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
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.signature import chop
    >>> from ndtorch.series import series
    >>> from ndtorch.series import clean
    >>> from ndtorch.taylor import taylor
    >>> def h(x):
    ...     q, p = x
    ...     h1 = q**3 + q**2*p + q*p**2 + p**3
    ...     h2 = q**4 + q**3*p + q**2*p**2 + q*p**3 + p**4
    ...     return h1 + h2
    >>> x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> t = derivative(3, lambda x: taylor(2, 1.0, h, x), x)
    >>> h = hamiltonian((3, ), x, [], t)
    >>> chop(h)
    >>> clean(series((2, ), (4, ), h))
    (3, 0): tensor([1.], dtype=torch.float64),
    (2, 1): tensor([1.], dtype=torch.float64),
    (1, 2): tensor([1.], dtype=torch.float64),
    (0, 3): tensor([1.], dtype=torch.float64),
    (4, 0): tensor([1.], dtype=torch.float64),
    (3, 1): tensor([1.], dtype=torch.float64),
    (2, 2): tensor([1.], dtype=torch.float64),
    (1, 3): tensor([1.], dtype=torch.float64),
    (0, 4): tensor([1.], dtype=torch.float64)}

    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.signature import chop
    >>> from ndtorch.series import series
    >>> from ndtorch.series import clean
    >>> from ndtorch.taylor import taylor
    >>> def h(x, k):
    ...     q, p = x
    ...     a, b = k
    ...     h1 = (1 + a)*q**3 + q**2*p + q*p**2 + (1 - a)*p**3
    ...     h2 = (1 + b)*q**4 + q**3*p + q**2*p**2 + q*p**3 + (1 - b)*p**4
    ...     return h1 + h2
    >>> x = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> k = torch.tensor([0.0, 0.0], dtype=torch.float64)
    >>> t = derivative((3, 1), lambda x, k: taylor(2, 1.0, h, x, k), x, k)
    >>> h = hamiltonian((3, 1), x, [k], t)
    >>> chop(h)
    >>> clean(series((2, 2), (4, 1), h))
    {(3, 0, 0, 0): tensor([1.], dtype=torch.float64),
     (2, 1, 0, 0): tensor([1.], dtype=torch.float64),
     (1, 2, 0, 0): tensor([1.], dtype=torch.float64),
     (0, 3, 0, 0): tensor([1.], dtype=torch.float64),
     (3, 0, 1, 0): tensor([1.], dtype=torch.float64),
     (0, 3, 1, 0): tensor([-1.], dtype=torch.float64),
     (4, 0, 0, 0): tensor([1.], dtype=torch.float64),
     (3, 1, 0, 0): tensor([1.], dtype=torch.float64),
     (2, 2, 0, 0): tensor([1.], dtype=torch.float64),
     (1, 3, 0, 0): tensor([1.], dtype=torch.float64),
     (0, 4, 0, 0): tensor([1.], dtype=torch.float64),
     (4, 0, 0, 1): tensor([1.], dtype=torch.float64),
     (0, 4, 0, 1): tensor([-1.], dtype=torch.float64)}

    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.signature import chop
    >>> from ndtorch.evaluate import evaluate
    >>> from ndtorch.propagate import identity
    >>> from ndtorch.propagate import propagate
    >>> from ndtorch.series import series
    >>> from ndtorch.series import clean
    >>> from ndtorch.taylor import taylor
    >>> from ndtorch.inverse import inverse
    >>> def fn(x, k, l, n=1):
    ...     (qx, px, qy, py), (k, ), l = x, k, l/(2.0*n)
    ...     for _ in range(n):
    ...         qx, qy = qx + l*px, qy + l*py
    ...         px, py = px - 1.0*l*k*(qx**2 - qy**2), py + 2.0*l*k*qx*qy
    ...         qx, qy = qx + l*px, qy + l*py
    ...     return torch.stack([qx, px, qy, py])
    >>> x = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    >>> k = torch.tensor([0.0], dtype=torch.float64)
    >>> t = identity((2, 1), [x, k])
    >>> t = propagate((4, 1), (2, 1), t, [k], fn, 0.1)
    >>> derivative(1, lambda x, k: evaluate(t, [x, k]), x, k, intermediate=False)
    tensor([[1.0000, 0.1000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.1000],
            [0.0000, 0.0000, 0.0000, 1.0000]], dtype=torch.float64)
    >>> l = derivative(1, lambda x, k: evaluate(t, [x, k]), x, k)
    >>> i = inverse(1, x, [k], l)
    >>> t = propagate((4, 1), (2, 1), i, [k], t)
    >>> chop(t)
    >>> derivative(1, lambda x, k: evaluate(t, [x, k]), x, k, intermediate=False)
    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]], dtype=torch.float64)
    >>> h = hamiltonian((2, 1), x, [k], t)
    >>> chop(h)
    >>> clean(series((4, 1), (3, 1), h))
    {(3, 0, 0, 0, 1): tensor([0.0167], dtype=torch.float64),
    (2, 1, 0, 0, 1): tensor([-0.0025], dtype=torch.float64),
    (1, 2, 0, 0, 1): tensor([0.0001], dtype=torch.float64),
    (1, 0, 2, 0, 1): tensor([-0.0500], dtype=torch.float64),
    (1, 0, 1, 1, 1): tensor([0.0050], dtype=torch.float64),
    (1, 0, 0, 2, 1): tensor([-0.0001], dtype=torch.float64),
    (0, 3, 0, 0, 1): tensor([-2.0833e-06], dtype=torch.float64),
    (0, 1, 2, 0, 1): tensor([0.0025], dtype=torch.float64),
    (0, 1, 1, 1, 1): tensor([-0.0003], dtype=torch.float64),
    (0, 1, 0, 2, 1): tensor([6.2500e-06], dtype=torch.float64)}
    >>> dx = torch.tensor([0.1, 0.01, 0.05, 0.01], dtype=torch.float64)
    >>> dk = torch.tensor([1.0], dtype=torch.float64)
    >>> fn(x + dx, k + dk, 0.1)
    tensor([0.1010, 0.0096, 0.0510, 0.0105], dtype=torch.float64)
    >>> taylor(1, 1.0, lambda x, k: evaluate(h, [x, k]), evaluate(l, [dx, dk]), dk)
    tensor([0.1010, 0.0096, 0.0510, 0.0105], dtype=torch.float64)

    Note
    ----
    Generator is equal to negative hamiltonian t = exp(-[h]) = exp([g])

    """
    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.lstsq(matrix, vector.unsqueeze(1)).solution.squeeze()

    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    n, *ns = order

    point = [state, *knobs]

    def auxiliary(*point) -> State:
        state, *_ = point
        return torch.zeros_like(state).sum()

    ht = derivative((n + 1, *ns), auxiliary, state, *knobs)

    def hf(*point):
        return evaluate(ht, [*point])

    def hs(*point):
        return taylor(count, 1.0, hf, *point)

    def objective(values, index, sequence, shape, unique):
        n, *ns = index
        for key, value in zip(unique, values):
            unique[key] = value
        value = build(sequence, shape, unique)
        set(ht, index, value)
        return derivative((n - 1, *ns),
                          hs,
                          *point,
                          intermediate=False,
                          jacobian=jacobian).flatten()

    dimension = (len(state), *(len(knob) for knob in knobs))

    start = start if start is not None else 3
    alter = not count
    count = count if count is not None else n - 1
    array = signature(ht)

    for i in array:
        n, *ns = i
        if n < start:
            set(ht, i, [])
            continue
        if alter:
            count = n - 2
        guess = get(ht, i)
        sequence, shape, unique = reduce(dimension, i, guess)
        guess = torch.stack([*unique.values()])
        vector, matrix = derivative(1,
                                    objective,
                                    guess,
                                    i,
                                    sequence,
                                    shape,
                                    unique,
                                    intermediate=True,
                                    jacobian=jacobian)
        tensor = get(table, (n - 1, *ns))
        tensor = tensor.flatten() if isinstance(tensor, Tensor) else torch.zeros_like(vector)
        values = solve(matrix, tensor - vector)
        for key, value in zip(unique, values):
            unique[key] = value
        set(ht, i, build(sequence, shape, unique))

    return ht


def solution(order:tuple[int],
             state:State,
             knobs:Knobs,
             hamiltonian: Table, *,
             count:Optional[int]=None,
             inverse:bool=False,
             jacobian:Optional[Callable]=None) -> Table:
    """
    Compute table solution for a given near identity hamiltonian table

    Parameters
    ----------
    order: tuple[int, ...]
        table order
    state: State
        state
    knobs: Knobs
        knobs
    hamiltonian: Table
        hamiltonian table representation
    count: Optional[int]
        number of terms to use in expansion (derived from input order if None)
    inverse: bool, default=False
        flag to inverse time direction
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Table

    Examples
    --------
    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.evaluate import evaluate
    >>> from ndtorch.evaluate import compare
    >>> from ndtorch.propagate import identity
    >>> from ndtorch.propagate import propagate
    >>> from ndtorch.inverse import inverse
    >>> from ndtorch.factorization import hamiltonian
    ... def mapping(x, k, l):
    ...     (qx, px, qy, py), (k, ), l = x, k, l/2
    ...     qx, qy = qx + l*px, qy + l*py
    ...     px, py = px - 1.0*l*k*(qx**2 - qy**2), py + 2.0*l*k*qx*qy
    ...     qx, qy = qx + l*px, qy + l*py
    ...     return torch.stack([qx, px, qy, py])
    >>> x = torch.tensor(4*[0.0], dtype=torch.float64)
    >>> k = torch.tensor(1*[0.0], dtype=torch.float64)
    >>> t = identity((2, 1), [x, k])
    >>> t = propagate((4, 1), (2, 1), t, [k], mapping, 0.1)
    >>> l = derivative(1, lambda x, k: evaluate(t, [x, k]), x, k)
    >>> t = propagate((4, 1), (2, 1), inverse(1, x, [k], l), [k], t)
    >>> h = hamiltonian((2, 1), x, [k], t)
    >>> compare(t, solution((2, 1), x, [k], h))
    True

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    n, *_ = order

    point = [state, *knobs]

    count = count if count is not None else (n - 1)

    def hf(*point):
        return evaluate(hamiltonian, [*point])

    def hs(*point, order=first(order)):
        return taylor(count, (-1.0)**inverse, hf, *point)

    return derivative(order, hs, point, jacobian=jacobian)


def hamiltonian_inverse(order:tuple[int, ...],
                        state:State,
                        knobs:Knobs,
                        data:Table, *,
                        start:Optional[int]=None,
                        count:Optional[int]=None,
                        solve:Optional[Callable]=None,
                        jacobian:Optional[Callable]=None) -> Table:
    """
    Compute near identity table inverse using single exponent representation

    Parameters
    ----------
    order: tuple[int, ...]
        table order
    state: State
        state
    knobs: Knobs
        knobs
    data: Table
        input near identity table
    start: Optional[int]
        starting order (degree)
    count: Optional[int]
        number of terms to use in expansion (derived from input order if None)
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
    >>> from ndtorch.derivative import derivative
    >>> from ndtorch.evaluate import evaluate
    >>> from ndtorch.evaluate import compare
    >>> from ndtorch.propagate import identity
    >>> from ndtorch.propagate import propagate
    >>> from ndtorch.inverse import inverse
    >>> def mapping(x, k, l):
    ...     (qx, px, qy, py), (k, ), l = x, k, l/2
    ...     qx, qy = qx + l*px, qy + l*py
    ...     px, py = px - 1.0*l*k*(qx**2 - qy**2), py + 2.0*l*k*qx*qy
    ...     qx, qy = qx + l*px, qy + l*py
    ...     return torch.stack([qx, px, qy, py])
    >>> x = torch.tensor(4*[0.0], dtype=torch.float64)
    >>> k = torch.tensor(1*[0.0], dtype=torch.float64)
    >>> t = identity((2, 1), [x, k])
    >>> t = propagate((4, 1), (2, 1), t, [k], mapping, 0.1)
    >>> l = derivative(1, lambda x, k: evaluate(t, [x, k]), x, k)
    >>> t = propagate((4, 1), (2, 1), inverse(1, x, [k], l), [k], t)
    >>> compare(inverse((2, 1), x, [k], t), hamiltonian_inverse((2, 1), x, [k], t))
    True

    """
    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.lstsq(matrix, vector.unsqueeze(1)).solution.squeeze()

    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    ht = hamiltonian((2, 1),
                     state,
                     [knobs],
                     data,
                     start=start,
                     count=count,
                     solve=solve,
                     jacobian=jacobian)

    return solution(order, state, knobs, ht, count=count, inverse=True, jacobian=jacobian)
