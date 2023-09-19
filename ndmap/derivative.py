"""
Derivative
----------

Computation of higher order derivatives

"""

from typing import TypeAlias
from typing import Callable
from typing import Optional
from typing import Union

from multimethod import multimethod

import torch
from torch import Tensor

from ndmap.util import flatten


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
def derivative(order:int,
               function:Callable,
               *args:tuple,
               intermediate:bool=True,
               jacobian:Optional[Callable]=None) -> Union[Table, Tensor]:
    """
    Compute function derivatives with respet to the first function argument upto a given order

    Note, if intermediate flag is ``False``, only the highest order derivative is returned

    Input function is expected to return a tensor or a (nested) list of tensors
    The first function argument is expected to be a tensor

    If the input function returns a tensor, output is called derivative table representation
    [value, jacobian, hessian, ...]
    The first argument is called an evaluation point (State)

    Parameters
    ----------
    order: int, non-negative
        maximum derivative order
    function: Callable
        input function
    *args: tuple
        input function arguments
    intermediate: bool, default=True
        flag to return intermediate derivatives
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Union[Table, Tensor]
        derivative table representation | function derivatives

    Examples
    --------
    >>> import torch
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> derivative(1, fn, x, y, intermediate=True)
    [tensor(0.), tensor([1., 1.])]

    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

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

    _, ys = local(*args)

    return list(flatten(ys, target=tuple))


@multimethod
def derivative(order:tuple[int, ...],
               function:Callable,
               *args:tuple,
               intermediate:bool=True,
               jacobian:Optional[Callable]=None) -> Union[Table, Tensor]:
    """
    Compute function derivatives with respet to several first function arguments upto given orders

    Note, if intermediate flag is ``False``, only the highest (total) order derivative is returned

    Input function is expected to return a tensor or a (nested) list of tensors
    The first several function arguments are expected to be tensors

    If the input function returns a tensor, output is called derivative table representation
    List of several first arguments is called an evaluation point (``Point``)

    Parameters
    ----------
    order: tuple[int, ...], non-negative
        maximum derivative orders
    function: Callable
        input function
    *args:
        input function arguments
    intermediate: bool, default=True
        flag to return intermediate derivatives
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev

    Returns
    -------
    Union[Table, Tensor]
        derivative table representation | function derivatives

    Examples
    --------
    >>> import torch
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> derivative((1, 1), fn, x, y)
    [[tensor(0.), tensor([0., 0.])], [tensor([1., 1.]), tensor([[1., 1.], [1., 1.]])]]
    
    """
    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    pars = [*args][len(order):]

    def fixed(*args):
        return function(*args, *pars)

    def build(order, value):
        def local(*args):
            return derivative(order,
                              lambda x: fixed(*args, x),
                              value,
                              intermediate=intermediate,
                              jacobian=jacobian)
        return local

    (order, value), *rest = zip(order, args)
    for degree, tensor in reversed(rest):
        def build(order, value, build=build(degree, tensor)):
            def local(*args):
                return derivative(order,
                                  lambda x: build(*args, x),
                                  value,
                                  intermediate=intermediate,
                                  jacobian=jacobian)
            return local

    return build(order, value)()


@multimethod
def derivative(order:int,
               function:Callable,
               point:Point,
               *pars:tuple,
               **kwargs:dict) -> Union[Table, Tensor]:
    """ Compute function derivatives at evaluation point upto a given order """
    return derivative(order, function, *point, *pars, **kwargs)


@multimethod
def derivative(order:tuple[int, ...],
               function:Callable,
               point:Point,
               *pars:tuple,
               **kwargs:dict) -> Union[Table, Tensor]:
    """ Compute function derivatives at evaluation point upto given orders """
    return derivative(order, function, *point, *pars, **kwargs)


@multimethod
def derivative(order:int,
               function:Callable,
               state:State,
               knobs:Knobs,
               *pars:tuple,
               **kwargs:dict) -> Union[Table, Tensor]:
    """ Compute function derivatives for state and knobs upto a given order wrt state """
    return derivative(order, function, state, *knobs, *pars, **kwargs)


@multimethod
def derivative(order:tuple[int, ...],
               function:Callable,
               state:State,
               knobs:Knobs,
               *pars:tuple,
               **kwargs:dict) -> Union[Table, Tensor]:
    """ Compute function derivatives for state and knobs upto given orders wrt state and knobs """
    return derivative(order, function, state, *knobs, *pars, **kwargs)
