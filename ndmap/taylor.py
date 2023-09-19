"""
Taylor
------

Taylor integration step

"""

from typing import TypeAlias
from typing import Callable
from typing import Union

from math import factorial

import torch
from torch import Tensor

from ndmap.bracket import bracket


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


def taylor(degree:int,
           length:Tensor,
           hamiltonian:Hamiltonian,
           state:State,
           *args:tuple) -> State:
    """
    Perform Taylor intergation step.

    Parameters
    ----------
    degree: int, non-negative
        integration degree
    length: Tensor
        integration step length
    hamiltonian: Hamiltonian
        (autonomous) Hamiltonian function
    state: State
        intial state
    *args: tuple
        additional arguments passed to (autonomous) Hamiltonian function

    Returns
    -------
    State

    Examples
    --------
    >>> from math import pi
    >>> import torch
    >>> t = torch.tensor(0.1)
    >>> f = torch.tensor(2.0*pi*0.1234)
    >>> q, p = torch.tensor([0.1, 0.1])
    >>> torch.stack([q*(f*t).cos() + p*(f*t).sin(), p*(f*t).cos() - q*(f*t).sin()])
    tensor([0.1074, 0.0920])
    >>> state = torch.stack([q, p])
    >>> knobs = torch.stack([f])
    >>> def h(state, knobs):
    ...    (q, p), (omega, ) = state, knobs
    ...    return omega*(q**2 + p**2)/2
    >>> [taylor(i, t, h, state, knobs) for i in range(4)]
    [tensor([0.1000, 0.1000]),
     tensor([0.1078, 0.0922]),
     tensor([0.1075, 0.0919]),
     tensor([0.1074, 0.0920])]

    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> x = torch.tensor([1.0, 1.0])
    >>> l = torch.tensor(0.1)
    >>> def h(x):
    ...    q, p = x
    ...    return p**2/2
    >>> derivative(1, lambda l: taylor(1, l, h, x), l)
    [tensor([1.1000, 1.0000]), tensor([1., 0.])]
    >>> derivative(1, lambda x: taylor(1, l, h, x), x)
    [tensor([1.1000, 1.0000]),
     tensor([[1.0000, 0.1000],
             [0.0000, 1.0000]])]
    derivative((1, 1), lambda l, x: taylor(1, l, h, x), l, x)
    [[tensor([1.1000, 1.0000]),
      tensor([[1.0000, 0.1000],
              [0.0000, 1.0000]])],
     [tensor([1., 0.]),
      tensor([[-0., 1.],
              [-0., -0.]])]]

    """
    local = torch.clone(state)

    def value(state, *args):
        return state

    for i in range(degree):
        value = bracket(hamiltonian, value)
        local += 1/factorial(i + 1)*(-length)**(i + 1)*value(state, *args)

    return local
