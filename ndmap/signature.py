"""
Signature
---------

Derivative table representation utilities

"""
from math import factorial

from typing import TypeAlias
from typing import Callable
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
def signature(order:tuple[int, ...], *,
              factor:bool=False) -> Signature:
    """
    Compute derivative table bottom elements signatures from given orders

    Parameters
    ----------
    order: tuple[int, ...]
        table order
    fator: bool, default=True
        flag to return elements multipliation factors

    Returns
    -------
    Signature
        bottom table elements signatures

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> signature((2, 1))
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    >>> signature((2, 1), factor=True)
    [((0, 0), 1.0),
     ((0, 1), 1.0),
     ((1, 0), 1.0),
     ((1, 1), 1.0),
     ((2, 0), 0.5),
     ((2, 1), 0.5)]
    
    """
    length = len(order)

    def generate(order, local, indices):
        if length == len(local):
            indices.append(tuple(local))
        else:
            for i in range(order[len(local)] + 1):
                generate(order, local + [i], indices)

    indices = []
    generate(order, [], indices)

    if not factor:
        return indices

    result = []
    for index in indices:
        value = 1.0
        for count in index:
            value *= 1.0/factorial(count)
        result.append((index, value))

    return result

@multimethod
def signature(table:Table, *,
              factor:bool=False) -> Signature:
    """
    Compute derivative table bottom elements signatures

    Note, signature elements corresponds to the bottom elements of a flattened derivative table
    Bottom element signature is a tuple integers, derivative orders with respect to each tensor
    Optionaly return elements multiplication factors
    Given a signature (n, m, ...), corresponding multiplication factor is 1/n! * 1/m! * ...

    Parameters
    ----------
    table: Table
        input derivative table representation
    fator: bool, default=True
        flag to return elements multipliation factors

    Returns
    -------
    Signature
        bottom table elements signatures

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((2, 1), fn, x, y)
    >>> signature(t)
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    >>> signature(t, factor=True)
    [((0, 0), 1.0),
     ((0, 1), 1.0),
     ((1, 0), 1.0),
     ((1, 1), 1.0),
     ((2, 0), 0.5),
     ((2, 1), 0.5)]
    
    """
    array = [signature([i], subtable, factor=factor) for i, subtable in enumerate(table)]
    return [*flatten(array, target=list)]


@multimethod
def signature(index:list[int],
              table:Table, *,
              factor:bool=False):
    """ (auxiliary) """
    return [signature(index + [i], subtable, factor=factor) for i, subtable in enumerate(table)]


@multimethod
def signature(index:list[int],
              table:Tensor, *,
              factor:bool=False):
    """ (auxiliary) """
    value = 1.0
    for count in index:
        value *= 1.0/factorial(count)
    return tuple(index) if not factor else (tuple(index), value)


def get(table:Table,
        index:tuple[int, ...]) -> Union[Tensor, Table]:
    """
    Get derivative table element at a given (bottom) element signature

    Note, index can correspond to a bottom element or a subtable

    Parameters
    ----------
    table: Table
        input derivative table representation
    index: tuple[int, ...]
        element signature

    Returns
    -------
    Union[Tensor, Table]
        element value

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((2, 1), fn, x, y)
    >>> get(t, (1, 1))
    tensor([[1., 1.],
        [1., 1.]])

    """
    if isinstance(index, int):
        return table[index]

    *ns, n = index
    for i in ns:
        table = table[i]
    return table[n]


def set(table:Table,
        index:tuple[int, ...],
        value:Union[Tensor, Table]) -> None:
    """
    Set derivative table element at a given (bottom) element signature

    Note, index can correspond to a bottom element or a subtable

    Parameters
    ----------
    table: Table
        input derivative table representation
    index: tuple[int, ...]
        element signature
    value: Union[Tensor, Table]
        element value

    Returns
    -------
    None

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((2, 1), fn, x, y)
    >>> set(t, (1, 1), 1 + get(t, (1, 1)))
    >>> get(t, (1, 1))
    tensor([[2., 2.],
            [2., 2.]])

    """
    if isinstance(index, int):
        table[index] = value
        return

    *ns, n = index
    for i in ns:
        table = table[i]
    table[n] = value


@multimethod
def apply(table:Table,
          index:tuple[int, ...],
          function:Callable) -> None:
    """
    Apply function (modifies element at index)

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
    -------
    None

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((2, 1), fn, x, y)
    >>> apply(t, (1, 1), torch.log)
    >>> get(t, (1, 1))
    tensor([[0., 0.],
            [0., 0.]])

    """
    value = get(table, index)
    set(table, index, function(value))


@multimethod
def apply(table:Table,
          index:list[tuple[int, ...]],
          function:Callable) -> None:
    """
    Apply function (modifies element at list of indices)

    Parameters
    ----------
    table: Table
        input derivative table representation
    index: int[tuple[int, ...]]
        list of element signatures
    function: Callable
        function to apply

    Returns
    -------
    None

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((2, 1), fn, x, y)
    >>> apply(t, [(1, 0), (1, 1)], torch.log)
    >>> get(t, [(1)])
    [tensor([0., 0.]),
    tensor([[0., 0.],
            [0., 0.]])]

    """
    for i in index:
        apply(table, i, function)


@multimethod
def apply(table:Table,
          function:Callable) -> None:
    """
    Apply function to all bottom elements

    Parameters
    ----------
    table: Table
        input derivative table representation
    function: Callable
        function to apply

    Returns
    -------
    None

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((2, 1), fn, x, y)
    >>> apply(t, torch.norm)
    >>> t
    [[tensor(0.), tensor(0.)],
     [tensor(1.4142), tensor(2.)],
     [tensor(3.1623), tensor(4.4721)]]

    """
    apply(table, signature(table), function)


def chop(table:Table, *,
         threshold:float=1.0E-9,
         value:float=0.0,
         replace:bool=False) -> None:
    """
    Chop tensor elements in a table below a given threshold

    Parameters
    ----------
    table: Table
        input derivative table representation
    threshold: float, default=1.0E-9
        threshold value
    value: float, default=0.0
        set value
    replace: bool, default=False
        flag to replace zero tensors

    Returns
    -------
    None

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (x1 + x2 + x1**2 + x1*x2 + x2**2)*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> t = derivative((0, 1), fn, x, y)
    >>> apply(t, lambda x: x + 1.0E-10)
    >>> chop(t)
    >>> t
    [[tensor(0.), tensor([0., 0.])]]

    """
    def inner(tensor):
        tensor = tensor.clone()
        tensor[tensor.abs() < threshold] = value
        if replace and torch.allclose(tensor, torch.zeros_like(tensor)):
            tensor = []
        return tensor
    apply(table, inner)


def to(table:Table,
       *args:tuple,
       **kwargs:dict) -> None:
    """
    Perform dtype and/or device conversion for all bottom table element elements (tensors)

    Parameters
    ----------
    *args: tuple
        passed to torch.to function
    **kwargs: dict
        passed to torch.to function

    Returns
    -------
    None

    Examples
    --------
    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> x = torch.tensor(0.0)
    >>> y = torch.tensor(0.0)
    >>> t = derivative((1, 1), lambda x, y: x + y, x, y)
    >>> to(t, torch.float64)
    >>> t
    [[tensor(0., dtype=torch.float64), tensor(1., dtype=torch.float64)],
    [tensor(1., dtype=torch.float64), tensor(0., dtype=torch.float64)]]
        [[tensor(0.), tensor([0., 0.])]]

    """
    apply(table, lambda tensor: tensor.to(*args, **kwargs))
