"""
Yoshida
-------

Yoshida integrator (coefficients and step)

"""

from typing import TypeAlias
from typing import Callable
from typing import Union
from typing import Optional

from functools import reduce
from itertools import groupby

from multimethod import multimethod

import torch
from torch import Tensor

from ndmap.util import first
from ndmap.util import last


State       : TypeAlias = Tensor
Knobs       : TypeAlias = list[Tensor]
Point       : TypeAlias = list[Tensor]
Scalar      : TypeAlias = Tensor
Delta       : TypeAlias = list[Tensor]
Table       : TypeAlias = list
Series      : TypeAlias = dict[tuple[int, ...], Tensor]
Signature   : TypeAlias = Union[list[tuple[int, ...]], list[tuple[tuple[int, ...], float]]]
Mapping     : TypeAlias = Callable
Observable  : TypeAlias = Callable
Hamiltonian : TypeAlias = Callable


@multimethod
def coefficients(n:int) -> list[float]:
    """
    Generate Yoshida coefficients for a given order

    Note, difference order is two times (Yoshida) order

    Given a time-reversible step f(2n)(dt) with difference order 2n (and n Yoshida order)
    2(n+1) integration step can be constructed using Yoshida coefficients with order (n+1)
    x1, x2, x1 = coefficients(n + 1)
    f(2(n+1))(dt) = f(2n)(x1 dt) o f(2n)(x2 dt) o f(2n)(x1 dt)

    Parameters
    ----------
    n: int, positive
        Yoshida order

    Returns
    -------
    list[float]

    Examples
    --------
    >>> coefficients(1)
    (1.3512071919596578, -1.7024143839193153, 1.3512071919596578)

    >>> coefficients(2)
    (1.1746717580893635, -1.3493435161787270, 1.1746717580893635)

    >>> coefficients(3)
    (1.1161829393253857, -1.2323658786507714, 1.1161829393253857)

    """
    return [
        +1/(2 - 2**(1/(1 + 2*n))),
        -2**(1/(1 + 2*n))/(2 - 2**(1/(1 + 2*n))),
        +1/(2 - 2**(1/(1 + 2*n)))
    ]


@multimethod
def coefficients(n:int,
                 m:int) -> list[float]:
    """
    Generate Yoshida coefficients for given Yoshida n <= m orders

    Given a time-reversible integration step f(2(n-1))(dt)
    Construct coefficients x1, x2, ..., x2, x1, so that
    f(2m)(dt) = f(2(n-1))(x1 dt) o f(2(n-1))(x2 dt) o ... o f(2(n-1))(x2 dt) o f(2(n-1))(x1 dt)

    Parameters
    ----------
    n: int, non-negative
        start Yoshida order
    m: int, non-negative
        final Yoshida order

    Returns
    -------
    list[float]

    Examples
    --------
    >>> coefficients(0, 0)
    [1.0]

    >>> coefficients(0, 1)
    [1.3512071919596578, -1.7024143839193153, 1.3512071919596578]

    >>> coefficients(1, 1)
    [1.3512071919596578, -1.7024143839193153, 1.3512071919596578]

    >>> coefficients(1, 2)
    [1.5872249277222432,
     -1.999778097355123,
     1.5872249277222432,
     -1.8232426634848289,
     2.2971418107909303,
     -1.8232426634848289,
     1.5872249277222432,
     -1.999778097355123,
     1.5872249277222432]

    >>> coefficients(2, 2)
    [1.1746717580893635, -1.349343516178727, 1.1746717580893635]

    """
    return reduce(
        lambda xs, x: [xi*xsi for xi in x for xsi in xs],
        map(coefficients, range(n if n != 0 else 1, m + 1)),
        [1.0]
    )


@multimethod
def coefficients(l:int,
                 n:int,
                 m:int,
                 merge:bool) -> list[list[int], list[float]]:
    """
    Generate Yoshida coefficients multistep

    Given a set of symplectic mappings indexed as 0, 1, ..., (l - 1) and Yoshida n <= m orders
    Construct Yoshida coefficients (i1, x1), (i2, x2), ..., (i2, x2), (i1, x1)
    f(2m)(dt) = f(i1)(x1 dx) o f(i2)(x2 dx) o ... o f(i2)(x2 dx) o f(i1)(x1 dx)


    Parameters
    ----------
    l: int, positive
        number of mappings
    n: int, non-negative
        start Yoshida order
    m: int, non-negative
        final Yoshida order
    merge: bool
        flag to merge edge mappings (assume commuting)

    Returns
    -------
    list[list[int], list[float]]

    Examples
    --------
    >>> coefficients(1, 0, 0, True)
    [[0], [1.0]]

    >>> coefficients(1, 1, 1, True)
    [[0], [1.0]]

    >>> coefficients(1, 1, 1, False)
    [[0, 0, 0], [1.3512071919596578, -1.7024143839193153, 1.3512071919596578]]

    >>> coefficients(2, 0, 0, True)
    [[0, 1, 0], [0.5, 1.0, 0.5]]

    >>> coefficients(4, 0, 0, True)
    [[0, 1, 2, 3, 2, 1, 0], [0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5]]

    >>> coefficients(2, 1, 1, True)
    [[0, 1, 0, 1, 0, 1, 0],
     [0.6756035959798289,
      1.3512071919596578,
      -0.17560359597982877,
      -1.7024143839193153,
      -0.17560359597982877,
      1.3512071919596578,
      0.6756035959798289]]

    >>> coefficients(2, 1, 1, False)
    [[0, 1, 0, 0, 1, 0, 0, 1, 0],
     [0.6756035959798289,
      1.3512071919596578,
      0.6756035959798289,
      -0.8512071919596577,
      -1.7024143839193153,
      -0.8512071919596577,
      0.6756035959798289,
      1.3512071919596578,
      0.6756035959798289]]

    """
    ps = [[i, 0.5] for i in range(l - 1)]
    ps = ps + [[l - 1, 1.0]] + [*reversed(ps)]
    ns, vs = map(list, zip(*ps))
    cs = coefficients(n, m)
    ps = sum(([[n, v*c] for (n, v) in zip(ns, vs)] for c in cs), start = [])
    if merge:
        gs = groupby(ps, key=lambda x: first(x))
        ps = [reduce(lambda x, y: [first(x), last(x) + last(y)], g) for _, g in gs]
    return [*map(list, zip(*ps))]


def yoshida(n:int,
            m:int,
            merge:bool,
            mappings:list[Mapping],
            parameters:Optional[list[list]]=None) -> Mapping:
    """
    Construct Yoshida integration multistep

    Parameters
    ----------
    n: int, non-negative
        start Yoshida order
    m: int, non-negative
        final Yoshida order
    merge: bool
        flag to merge edge mappings (assume commuting)
    mappings: list[Mapping]
        list of (time-reversible) mappings
    parameters: Optional[list[list]], default=None
        list of optional fixed parameters for each mapping

    Returns
    -------
    Mapping

    Examples
    --------
    >>> def fn(x, t): q, p = x ; return torch.stack([q, p - t*q])
    >>> def gn(x, t): q, p = x ; return torch.stack([q + t*p, p])
    >>> t = torch.tensor(0.5, dtype=torch.float64)
    >>> x = torch.tensor([0.1, 0.1], dtype=torch.float64)
    >>> yoshida(1, 0, True, [fn, gn])(x, t)
    tensor([0.1344, 0.0375], dtype=torch.float64)
    >>> yoshida(1, 1, True, [fn, gn])(x, t)
    tensor([0.1358, 0.0402], dtype=torch.float64)
    >>> yoshida(1, 2, True, [fn, gn])(x, t)
    tensor([0.1357, 0.0398], dtype=torch.float64)
    >>> yoshida(1, 3, True, [fn, gn])(x, t)
    tensor([0.1357, 0.0398], dtype=torch.float64)
    >>> yoshida(1, 4, True, [fn, gn])(x, t)
    tensor([0.1357, 0.0398], dtype=torch.float64)

    >>> def fn(x, t): q, p = x ; return torch.stack([q, p - t*q])
    >>> def gn(x, t): q, p = x ; return torch.stack([q + t*p, p])
    >>> def s2(x, t):
    ...     x1, x2, x1 = 0.5, 1.0, 0.5
    ...     y = torch.clone(x)
    ...     x = fn(x, x1*t)
    ...     x = gn(x, x2*t)
    ...     x = fn(x, x1*t)
    ...     return x
    >>> def s4(x, t):
    ...     x1, x2, x1 = coefficients(1)
    ...     x = torch.clone(x)
    ...     x = s2(x, x1*t)
    ...     x = s2(x, x2*t)
    ...     x = s2(x, x1*t)
    ...     return x
    >>> def s6(x, t):
    ...     x1, x2, x1 = coefficients(2)
    ...     x = torch.clone(x)
    ...     x = s4(x, x1*t)
    ...     x = s4(x, x2*t)
    ...     x = s4(x, x1*t)
    ...     return x
    >>> t = torch.tensor(0.5, dtype=torch.float64)
    >>> x = torch.tensor([0.1, 0.1], dtype=torch.float64)
    >>> torch.allclose(s2(x, t), yoshida(0, 0, True, [fn, gn])(x, t))
    True
    >>> torch.allclose(s4(x, t), yoshida(0, 1, True, [fn, gn])(x, t))
    True
    >>> torch.allclose(s6(x, t), yoshida(0, 2, True, [fn, gn])(x, t))
    True
    >>> torch.allclose(yoshida(1, 1, False, [s2])(x, t), yoshida(0, 1, True, [fn, gn])(x, t))
    True
    >>> torch.allclose(yoshida(2, 2, False, [s4])(x, t), yoshida(0, 2, True, [fn, gn])(x, t))
    True
    >>> torch.allclose(yoshida(3, 3, False, [s6])(x, t), yoshida(0, 3, True, [fn, gn])(x, t))
    True

    >>> from ndmap.derivative import derivative
    >>> def fn(x, t, k): q, p = x ; return torch.stack([q, p - t*k*q])
    >>> def gn(x, t, k): q, p = x ; return torch.stack([q + t*p, p])
    >>> t = torch.tensor(0.5, dtype=torch.float64)
    >>> x = torch.tensor([0.1, 0.1], dtype=torch.float64)
    >>> k = torch.tensor(1.0, dtype=torch.float64)
    >>> derivative(1, yoshida(1, 2, True, [fn, gn]), x, t, k)
    [tensor([0.1357, 0.0398], dtype=torch.float64),
     tensor([[ 0.8775,  0.4800],
             [-0.4792,  0.8775]], dtype=torch.float64)]
    >>> derivative((1, 1), yoshida(1, 2, True, [fn, gn]), x, t, k)
    [[tensor([0.1357, 0.0398], dtype=torch.float64),
      tensor([ 0.0404, -0.1356], dtype=torch.float64)],
     [tensor([[ 0.8775,  0.4800],
              [-0.4792,  0.8775]], dtype=torch.float64),
      tensor([[-0.4810,  0.8850],
              [-0.8750, -0.4810]], dtype=torch.float64)]]
    >>> derivative((1, 1, 1), yoshida(1, 2, True, [fn, gn]), x, t, k)
    [[[tensor([0.1357, 0.0398], dtype=torch.float64),
       tensor([-0.0139, -0.0579], dtype=torch.float64)],
      [tensor([ 0.0404, -0.1356], dtype=torch.float64),
       tensor([-0.0563, -0.1213], dtype=torch.float64)]],
     [[tensor([[ 0.8775,  0.4800],
               [-0.4792,  0.8775]], dtype=torch.float64),
       tensor([[-0.1202, -0.0187],
               [-0.4584, -0.1202]], dtype=torch.float64)],
      [tensor([[-0.4810,  0.8850],
               [-0.8750, -0.4810]], dtype=torch.float64),
       tensor([[-0.4654, -0.0978],
               [-0.7473, -0.4654]], dtype=torch.float64)]]]

    >>> from ndmap.propagate import identity
    >>> from ndmap.propagate import propagate
    >>> from ndmap.derivative import derivative
    >>> def fn(x, t, k): q, p = x ; return torch.stack([q, p - t*k*q])
    >>> def gn(x, t, k): q, p = x ; return torch.stack([q + t*p, p])
    >>> t = torch.tensor(0.5, dtype=torch.float64)
    >>> x = torch.tensor([0.1, 0.1], dtype=torch.float64)
    >>> k = torch.tensor(1.0, dtype=torch.float64)
    >>> propagate((2, ), (1, ), identity((1, ), [x]), [], yoshida(1, 2, True, [fn, gn]), t, k)
    [tensor([0.1357, 0.0398], dtype=torch.float64),
     tensor([[ 0.8775,  0.4800],
             [-0.4792,  0.8775]], dtype=torch.float64)]

    Note
    ----
    Each signature mapping in mappings is (state, delta, *args)
    Fixed parameters are passed at the end
    Coefficients are attached to the output (table attribute)

    """
    table = coefficients(len(mappings), n, m, merge)

    parameters = [[] for _ in range(len(mappings))] if parameters is None else parameters
    parameters = [parameters[i] for i in first(table)]

    def closure(state:State, delta:Tensor, *args:tuple) -> State:
        local = torch.clone(state)
        for index, value in zip(*table):
            local = mappings[index](local, value*delta, *args, *parameters[index])
        return local

    closure.table = table
    closure.parameters = parameters

    return closure
