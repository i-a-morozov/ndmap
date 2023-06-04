"""
Bracket
-------

Poisson bracket computation

"""

from typing import TypeAlias
from typing import Callable
from typing import Union
from typing import Optional

from multimethod import multimethod

import torch
from torch import Tensor

from .util import symplectic
from .derivative import derivative
from .evaluate import evaluate


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
def bracket(f:Union[Observable, Mapping],
            g:Union[Observable, Mapping], *,
            jacobian:Optional[Callable]=None) -> Union[Observable, Mapping, Callable]:
    """
    Compute Poisson bracket

    Parameters
    ----------
    f, g: Union[Observable, Mapping]
        f, g
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Union[Observable, Mapping, Callable]

    Examples
    --------
    >>> import torch
    >>> def f(state): q, p = state; return q
    >>> def g(state): q, p = state; return p
    >>> state = torch.tensor([0.0, 0.0])
    >>> bracket(f, g)(state)
    tensor(1.)

    >>> import torch
    >>> def f(state): q, p = state; return (q**2 + p**2)/2
    >>> def g(state): return state
    >>> state = torch.tensor([1.0, 1.0])
    >>> bracket(f, g)(state)
    tensor([-1.,  1.])

    >>> import torch
    >>> def f(state): return state
    >>> def g(state): return state
    >>> state = torch.tensor([0.0, 0.0, 0.0, 0.0])
    >>> bracket(f, g)(state)
    tensor([[ 0.,  1.,  0.,  0.],
            [-1.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  1.],
            [ 0.,  0., -1.,  0.]])

    >>> import torch
    >>> from ndtorch.derivative import derivative
    >>> def f(state): q, p = state; return -p**2/2
    >>> def g(state): return state
    >>> state = torch.tensor([0.0, 0.0])
    >>> derivative(1, bracket(f, g), state, intermediate=False)
    tensor([[0., 1.],
            [0., 0.]])

    Note
    ----
    (Observable, Observable)                       -> Observable
    (Observable, Mapping) or (Mapping, Observable) -> Mapping
    (Mapping, Mapping)                             -> Callable

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    def closure(state: State, *args:tuple) -> Union[Scalar, State, Tensor]:
        df = derivative(1, f, state, *args, intermediate=False, jacobian=jacobian)
        dg = derivative(1, g, state, *args, intermediate=False, jacobian=jacobian)
        return df @ symplectic(state) @ dg

    return closure


@multimethod
def bracket(tf:Union[Table, Series],
            tg:Union[Table, Series], *,
            jacobian:Optional[Callable]=None) -> Union[Observable, Mapping, Callable]:
    """
    Compute Poisson bracket

    Parameters
    ----------
    tf, tg: Union[Table, Series]
        f, g table or series representation
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Union[Observable, Mapping, Callable]

    Examples
    --------
    >>> import torch
    >>> from ndtorch.propagate import identity
    >>> from ndtorch.propagate import propagate
    >>> def f(x):
    ...     x1, x2 = x
    ...     return x1**2
    >>> def g(x):
    ...     x1, x2 = x
    ...     return x2**2
    >>> state = torch.tensor([1.0, 1.0], dtype=torch.float64)
    >>> delta = torch.zeros_like(state)
    >>> tf = identity((2, ), [state])
    >>> tf = propagate((2, ), (2, ), tf, [], f)
    >>> tg = identity((2, ), [state])
    >>> tg = propagate((2, ), (2, ), tg, [], g)
    >>> torch.allclose(bracket(tf, tg)(delta), bracket(f, g)(state))
    True

    >>> import torch
    >>> from ndtorch.util import curry_apply
    >>> from ndtorch.propagate import identity
    >>> from ndtorch.propagate import propagate
    >>> def f(x):
    ...     x1, x2 = x
    ...     return x1**2
    >>> def g(x):
    ...     x1, x2 = x
    ...     return x2**2
    >>> state = torch.tensor([1.0, 1.0], dtype=torch.float64)
    >>> delta = torch.zeros_like(state)
    >>> sf = identity((2, ), [state], flag=True)
    >>> sf = propagate((2, ), (2, ), sf, [], f)
    >>> sg = identity((2, ), [state], flag=True)
    >>> sg = propagate((2, ), (2, ), sg, [], g)
    >>> torch.allclose(bracket(sf, sg)(delta), bracket(f, g)(state))
    True

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    def f(state:State, *args:tuple) -> Union[Scalar, State, Tensor]:
        return evaluate(tf, [state, *args])

    def g(state:State, *args:tuple) -> Union[Scalar, State, Tensor]:
        return evaluate(tg, [state, *args])

    return bracket(f, g, jacobian=jacobian)