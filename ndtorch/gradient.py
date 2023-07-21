"""
Gradient
--------

Computation of higher order derivatives with torch gradient function

"""

from typing import Callable
from typing import Optional
from typing import Union

from math import factorial
from math import prod
from itertools import product
from functools import partial

import torch
from torch import Tensor


def signature(order:tuple[int, ...]) -> list[tuple[int, ...]]:
    """
    Compute derivative signatures from given total group orders

    Parameters
    ----------
    order: tuple[int, ...], non-negative
        tuple of orders

    Returns
    -------
    list[tuple[int, ...]]
        list of signatures

    Examples
    --------
    >>> signature((2, ))
    [(0,), (1,), (2,)]
    >>> signature((2, 2))
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    
    """
    def accumulate(order:tuple[int, ...],
                   container:list[int],
                   output:list[tuple[int, ...]]) -> None:
        count:int = len(container)
        if count == limit:
            output.append(tuple(container))
            return
        for i in range(1 + order[count]):
            accumulate(order, container + [i], output)
    limit:int = len(order)
    output:list[tuple[int, ...]] = []
    accumulate(order, [], output)
    return output


def split(array:tuple[int, ...],
          chunks:tuple[int, ...]) -> list[tuple[int, ...]]:
    """
    Split array into chuncks with given length

    Parameters
    ----------
    array: tuple[int, ...]
        array to split
    chunks: tuple[int, ...], positive
        list of chunks to use

    Returns
    -------
    list[tuple[int, ...]]
    
    Examples
    --------
    >>> split((1, 2, 3, 4, 5), (2, 3))
    [(1, 2), (3, 4, 5)]

    """
    marker:int = 0
    output:list[tuple[int, ...]] = []
    for chunk in chunks:
        output.append(array[marker:marker + chunk])
        marker += chunk
    return output


def group(arrays:list[tuple[int, ...]],
          dimension:tuple[int, ...]) -> list[tuple[int, ...]]:
    """
    Standard indices grouping

    Parameters
    ----------
    arrays: list[tuple[int, ...]], non-negative
        input indices
    dimension: tuple[int, ...], positive
        input dimension

    Returns
    -------
    list[tuple[int, ...]]

    Examples
    --------
    >>> xs = [(0, 2, 0, 1),(0, 2, 1, 0), (1, 1, 0, 1), (1, 1, 1, 0), (2, 0, 0, 1), (2, 0, 1, 0)]
    >>> group(xs, (2, 2))
    [(2, 0, 1, 0),
     (2, 0, 0, 1),
     (1, 1, 1, 0),
     (1, 1, 0, 1),
     (0, 2, 1, 0),
     (0, 2, 0, 1)]

    """
    output:list[list[tuple[int, ...]]] = [split(array, dimension) for array in arrays]
    output:list[tuple[int, ...]] = sorted(output, reverse=True, key=lambda x: [*map(sum, x)])
    return [*reversed([*map(lambda x: sum(x, start=()), output)])]


def index(dimension:tuple[int, ...],
          order:tuple[int, ...], *,
          group:Optional[Callable]=None,
          signature:Optional[list[tuple[int, ...]]]=None) -> list[tuple[int, ...]]:
    """
    Generate monomial index table for a given dimension and order

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        monomial dimensions
    order: tuple[int, ...], non-negative
        derivative orders (total monomial degrees)
    group: Optional[Callable]
        grouping function
    signature: Optional[list[tuple[int, ...]]]
        allowed signatures

    Returns
    -------
    list[tuple[int, ...]]
        monomial index table

    Examples
    --------
    >>> index((2, ), (2, ))
    [(0, 2), (1, 1), (2, 0)]
    >>> index((4, ), (3, )) == index((2, 2), (2, 1))
    True
    >>> index((2, 2), (2, 1), signature=signature((2, 1)))
    [(0, 2, 0, 1),
     (0, 2, 1, 0),
     (1, 1, 0, 1),
     (1, 1, 1, 0),
     (2, 0, 0, 1),
     (2, 0, 1, 0)] 
    >>> index((2, 2), (2, 1), signature=signature((2, 1)), group=group)
    [(2, 0, 1, 0),
     (2, 0, 0, 1),
     (1, 1, 1, 0),
     (1, 1, 0, 1),
     (0, 2, 1, 0),
     (0, 2, 0, 1)]

    """
    length:int = sum(dimension)
    degree:int = sum(order)
    output:list[tuple[int, ...]] = []
    for i in product(range(degree + 1), repeat=length):
        if sum(i) == degree:
            if signature is None:
                output.append(i)
            elif tuple(map(sum, split(i, dimension))) in signature:
                output.append(i)
    return group(output, dimension) if group else output


def factor(arrays:list[tuple[int, ...]]) -> list[float]:
    """
    Compute monomian factors given list of exponents

    Parameters
    ----------
    arrays: list[tuple[int, ...]], non-negative
        input indices

    Returns
    -------
    list[float]

    Examples
    --------
    >>> factor([(1, 0)])
    [1.0]
    >>> factor([(2, 0)])
    [0.5]
    >>> factor([(2, 2)])
    [0.25]
    >>> factor([(2, 2, 2)])
    [0.125]

    """
    return [*map(lambda x: 1.0 / prod(map(factorial, x)), arrays)]


def naught(state:Tensor) -> Tensor:
    """
    Infinitely differentiable zero function

    Parameters
    ----------
    state: Tensor
        input tensor

    Returns
    -------
    Tensor

    Examples
    --------
    >>> import torch
    >>> from torch.autograd import grad
    >>> x = torch.tensor(0.0, requires_grad=True)
    >>> y = (lambda x: x)(x)
    >>> y, *_ = grad(y, x, retain_graph=True, create_graph=True)
    >>> y
    tensor(1.)
    >>> y = (lambda x: x + naught(x))(x)
    >>> y, *_ = grad(y, x, retain_graph=True, create_graph=True)
    >>> y
    tensor(1., grad_fn=<AddBackward0>)
    >>> y, *_ = grad(y, x)
    >>> y
    tensor(-0.)

    """
    return 0 * torch.sin(state.sum())


def scalar(function:Callable,
           table:tuple[int]) -> Callable:
    """
    Given f(x, y, ...) and table = map(len, (x, y, ...)) return g(*x, *y, ...) = f(x, y, ...)

    Parameters
    ----------
    function: Callable
        input function
    table: tuple[int, ...]
        map(len, (x, y, ...))
    *pars: tuple
        passed to input function

    Returns
    -------
    Callable

    Examples
    --------
    >>> import torch
    >>> def fn(x, y):
    ...    x1, x2 = x
    ...    y1, y2, y3 = y
    ...    return x1*x2*y1*y2*y3
    >>> def gn(x1, x2, y1, y2, y3):
    ...    return fn((x1, x2), (y1, y2, y3))
    >>> x = torch.tensor([1, 1])
    >>> y = torch.tensor([1, 1, 1])
    >>> gn(*x, *y) == scalar(fn, (2, 3))(*x, *y)
    tensor(True)

    """
    def clouser(*args:tuple):
        return function(*[torch.stack(arg) for arg in split(args, table)])
    return partial(clouser)


def select(function:Callable,
           index:int, *,
           naught:Callable=naught) -> Callable:
    """
    Generate scalar function

    Parameters
    ----------
    function: Callable
        function
    index: int, non-negative
        index
    naught: Callable, default=naught
        zero function

    Returns
    -------
    Callable

    Examples
    --------
    >>> import torch
    >>> def fn(x1, x2, x3, x4):
    ...    return torch.stack([x1, x2, x3, x4])
    >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> [select(fn, i)(*x) for i in range(4)]
    [tensor(1.), tensor(2.), tensor(3.), tensor(4.)]

    Note
    ----
    Input fuction is assumed to have scalar tensor arguments

    """
    def clouser(*args:tuple):
        output:Tensor = function(*args).flatten()[index]
        return output + naught(torch.stack([*args])) if naught else output
    return clouser


def reduce(array:tuple[int, ...]) -> list[tuple[int, ...]]:
    """
    Generate direct path from given index to zero

    Parameters
    ----------
    array: tuple[int, ...], non-negative
        input index

    Returns
    -------
    list[tuple[int, ...]]

    Examples
    --------
    >>> reduce((4, 0))
    [(4, 0), (3, 0), (2, 0), (1, 0), (0, 0)]
    >>> reduce((2, 2))
    [(2, 2), (1, 2), (0, 2), (0, 1), (0, 0)]
    >>> reduce((2, 2, 2))
    [(2, 2, 2), (1, 2, 2), (0, 2, 2), (0, 1, 2), (0, 0, 2), (0, 0, 1), (0, 0, 0)]

    """
    output:list[tuple[int, ...]] = [array]
    count:int = 0
    while array:
        head, *tail = array
        if head != 0:
            array = tuple([head - 1, *tail])
            output.append(tuple(count*[0] + [*array]))
        else:
            array = tail
            count += 1
    return output


def series(order:tuple[int, ...],
           function:Callable,
           *args:tuple,
           retain:bool=True,
           series:bool=True,
           intermediate:Union[bool, tuple[int, ...]]=True,
           group:Callable=group,
           naught:Callable=naught,
           shape:Optional[tuple[int, ...]]=None) -> dict[tuple[int, ...], Tensor]:
    """
    Generate series representation of a given input function

    c(i, j, k, ...) * x**i * y**j * z**k * ... => {..., (i, j, k, ...) : c(i, j, k, ...), ...}

    Note, the input function (returns a tensor) arguments are expected to be vector tensors

    Parameters
    ----------
    order: tuple[int, ...], non-negative
        maximum derivative orders
    function: Callable
        input function
    *args: tuple
        input function arguments
    retain: bool, default=True
        flag to retain computation graph
    series: bool, default=True
        flag to return series coefficiens
    intermediate: Union[bool, tuple[int, ...]]
        flag to return indermidiate derivatives/coefficients
    group: Callable, default=group
        indices grouping function
    naught: Callable
        zero function
    shape: Optional[tuple[int, ...]]
        input function output shape

    Returns
    -------
    dict[tuple[int, ...], Tensor]

    Examples
    --------
    >>> import torch
    >>> def fn(x, y, a, b):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (a*(x1 + x2) + b*(x1**2 + x1*x2 + x2**2))*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> series((2, 1), fn, x, y, 1.0, 1.0, retain=False, series=True, intermediate=True)
    {(0, 0, 0, 0): tensor(0.),
     (0, 0, 1, 0): tensor(0.),
     (0, 0, 0, 1): tensor(0.),
     (1, 0, 0, 0): tensor(1.),
     (0, 1, 0, 0): tensor(1.),
     (1, 0, 1, 0): tensor(1.),
     (1, 0, 0, 1): tensor(1.),
     (0, 1, 1, 0): tensor(1.),
     (0, 1, 0, 1): tensor(1.),
     (2, 0, 0, 0): tensor(1.),
     (1, 1, 0, 0): tensor(1.),
     (0, 2, 0, 0): tensor(1.),
     (2, 0, 1, 0): tensor(1.),
     (2, 0, 0, 1): tensor(1.),
     (1, 1, 1, 0): tensor(1.),
     (1, 1, 0, 1): tensor(1.),
     (0, 2, 1, 0): tensor(1.),
     (0, 2, 0, 1): tensor(1.)}
    >>> series((2, 1), fn, x, y, 1.0, 1.0, retain=False, series=True, intermediate=False)
    {(2, 0, 1, 0): tensor(1.),
     (2, 0, 0, 1): tensor(1.),
     (1, 1, 1, 0): tensor(1.),
     (1, 1, 0, 1): tensor(1.),
     (0, 2, 1, 0): tensor(1.),
     (0, 2, 0, 1): tensor(1.)}
    >>> series((2, 1), fn, x, y, 1.0, 1.0, retain=False, series=True, intermediate=(2, 0, 1, 0))
    {(2, 0, 1, 0): tensor(1.)}
    >>> series((2, 1), fn, x, y, 1.0, 1.0, retain=False, series=False, intermediate=(2, 0, 1, 0))
    {(2, 0, 1, 0): tensor(2.)}
    >>> series((2, 1), fn, x, y, 1.0, 1.0, retain=True, series=True, intermediate=(2, 0, 1, 0))
    {(2, 0, 1, 0): tensor(1., grad_fn=<MulBackward0>)}

    """
    pars = [*args][len(order):]
    args = [*args][:len(order)]

    dimension:tuple[int, ...] = tuple(map(len, args))

    fixed:Callable = scalar(lambda *args: function(*args, *pars), dimension)

    array:list[tuple[int, ...]] = signature(order)
    table:dict[int, list[tuple[int, ...]]] = {}
    for i in range(1 + sum(order)):
        table[i] = index(dimension, (i, ), signature=array, group=group)

    if isinstance(intermediate, tuple):
        allow:list[tuple[int, ...]] = reduce(intermediate)
        for i in range(1 + sum(order)):
            table[i] = [array for array in table[i] if array in allow]

    units:list[tuple[int, ...]] = sorted(index(dimension, (1, )), reverse=True)
    units:dict[tuple[int, ...], int] = dict(zip(units, range(len(units))))

    bases:dict[tuple[int, ...], int] = {}
    bases[tuple(sum(dimension)*[0])] = 0

    shape:tuple[int, ...] = shape if shape else tuple(function(*args, *pars).shape)
    limit:int = prod(shape)

    state:Tensor = torch.cat(args)
    if not state.requires_grad:
        state.requires_grad = True
    state:list[Tensor] = [*state]

    values:list[Tensor] = []
    for i in range(limit):
        count:int = 0
        local:list[Tensor] = []
        local.append(select(fixed, i)(*state))
        for j in range(sum(order)):
            bank:list[tuple[int, ...]] = []
            for unit, case in units.items():
                unit:Tensor = torch.tensor(unit)
                arrays:Tensor = unit + torch.tensor(table[j])
                arrays:list[tuple[int, ...]] = [*map(tuple, arrays.tolist())]
                positions:list[int] = [*map(lambda x: bases[x], table[j])]
                for array, position in zip(arrays, positions):
                    if array not in bank and array in table[j + 1]:
                        bank.append(array)
                        y, *_ = torch.autograd.grad(local[position],
                                                    state[case],
                                                    create_graph=True,
                                                    retain_graph=True)
                        local.append(y)
                        count += 1
                        bases[array] = count
        values.append(torch.stack(local) if retain else torch.stack(local).detach())
    values:Tensor = torch.stack(values).T
    values:list[Tensor] = [*values.reshape(-1, *shape)]

    keys:list[tuple[int, ...]] = sum([*table.values()], start=[])

    if not intermediate or isinstance(intermediate, tuple):
        keys = [key for key in keys if tuple(map(sum, split(key, dimension))) == order]

    if series:
        coefficients:list[float] = factor(keys)
        return {key: coefficient*values[bases[key]] for key, coefficient in zip(keys, coefficients)}

    return {key: values[bases[key]] for key in keys}


def jacobian(function:Callable) -> Callable:
    """
    Compute function jacobian (can be composed)

    Note, the output shape is different from jacfwd or jacrev
    
    Parameters
    ----------
    function:Callable
        function

    Returns
    -------
    Callable

    Examples
    --------
    >>> import torch
    >>> def fn(x):
    ...    x1, x2 = x
    ...    return torch.stack([1.0*x1 + 2.0*x2, 3.0*x1 + 4.0*x2])
    >>> x = torch.tensor([0.0, 0.0])
    >>> torch.func.jacrev(fn)(x)
    tensor([[1., 2.],
            [3., 4.]])
    >>> jacobian(fn)(x).detach().permute(1, 0)
    tensor([[1., 2.],
            [3., 4.]])
            
    >>> import torch
    >>> def fn(x):
    ...    x1, x2 = x
    ...    y1 = torch.stack([1.0*x1 + 2.0*x2, 3.0*x1 + 4.0*x2])
    ...    y2 = torch.stack([5.0*x1 + 6.0*x2, 7.0*x1 + 8.0*x2])
    ...    return torch.stack([y1, y2])
    >>> x = torch.tensor([0.0, 0.0])
    >>> torch.func.jacrev(fn)(x)
    tensor([[[1., 2.],
             [3., 4.]],
            [[5., 6.],
             [7., 8.]]])
    >>> jacobian(fn)(x).detach().permute(1, 2, 0)
    tensor([[[1., 2.],
             [3., 4.]],
            [[5., 6.],
             [7., 8.]]])

    """
    def clouser(*args):
        arg, *args = args
        table:dict[tuple[int, ...], Tensor] = series((1, ), function, arg, *args)
        output:Tensor = torch.stack([table[key] for key in index((len(arg),), (1,), group=group)])
        return output
    return clouser


def hessian(function:Callable) -> Callable:
    """
    Compute function hessian

    Parameters
    ----------
    function:Callable
        function
        
    Returns
    -------
    Callable
    
    Examples
    --------
    >>> import torch
    >>> def fn(x):
    ...    x1, x2, x3, x4 = x
    ...    return 1.0*x1**2 + 2.0*x2**2+ 3.0*x3**2 + 4.0*x4**2
    >>> x = torch.tensor([0.0, 0.0, 0.0, 0.0])
    >>> torch.func.hessian(fn)(x)
    tensor([[2., 0., 0., 0.],
            [0., 4., 0., 0.],
            [0., 0., 6., 0.],
            [0., 0., 0., 8.]])
    >>> hessian(fn)(x).detach().permute(1, 0)
    tensor([[2., 0., 0., 0.],
            [0., 4., 0., 0.],
            [0., 0., 6., 0.],
            [0., 0., 0., 8.]])
    
    """
    return jacobian(jacobian(function))


def evaluate(series:dict[tuple[int, ...], Tensor],
             delta:list[Tensor]) -> Tensor:
    """
    Evaluate series

    Parameters
    ----------
    series: Series
        input series representation
    delta: list[Tensor]
        delta deviation

    Returns
    -------
    Tensor

    Examples
    --------
    >>> import torch
    >>> def fn(x, y, a, b):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (a*(x1 + x2) + b*(x1**2 + x1*x2 + x2**2))*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> s = series((2, 1), fn, x, y, 1.0, 1.0, retain=False)
    >>> dx = torch.tensor([1.0, 2.0])
    >>> dy = torch.tensor([3.0, 4.0])
    >>> fn(x + dx, y + dy, 1.0, 1.0)
    tensor(80.)
    >>> evaluate(s, [dx, dy])
    tensor(80.)

    >>> import torch
    >>> def fn(x, y, a, b):
    ...    x1, x2 = x
    ...    y1, y2 = y
    ...    return (a*(x1 + x2) + b*(x1**2 + x1*x2 + x2**2))*(1 + y1 + y2)
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.zeros_like(x)
    >>> s = series((2, 1), fn, x, y, 1.0, 1.0, retain=False)
    >>> series((2, 1), lambda x, y: evaluate(s, [x, y]), x, y, retain=False)
    {(0, 0, 0, 0): tensor(0.),
     (0, 0, 1, 0): tensor(0.),
     (0, 0, 0, 1): tensor(0.),
     (1, 0, 0, 0): tensor(1.),
     (0, 1, 0, 0): tensor(1.),
     (1, 0, 1, 0): tensor(1.),
     (1, 0, 0, 1): tensor(1.),
     (0, 1, 1, 0): tensor(1.),
     (0, 1, 0, 1): tensor(1.),
     (2, 0, 0, 0): tensor(1.),
     (1, 1, 0, 0): tensor(1.),
     (0, 2, 0, 0): tensor(1.),
     (2, 0, 1, 0): tensor(1.),
     (2, 0, 0, 1): tensor(1.),
     (1, 1, 1, 0): tensor(1.),
     (1, 1, 0, 1): tensor(1.),
     (0, 2, 1, 0): tensor(1.),
     (0, 2, 0, 1): tensor(1.)}

    """
    state:Tensor = torch.cat(delta)
    local:Tensor = torch.ones_like(state).prod()
    total:Tensor = torch.zeros_like(next(iter(series.values())))
    for key, value in series.items():
        for i, x in zip(key, state):
            for _ in range(i):
                local = x * local
        total = total + value * local
        local = 1.0
    return total


def derivative(series:dict[tuple[int, ...], Tensor],
               index:tuple[int, ...]) -> dict[tuple[int, ...], Tensor]:
    """
    Compute series derivative

    Parameters
    ----------
    series: dict[tuple[int, ...], Tensor]
        series
    index:
        derivative index

    Returns
    -------
    dict[tuple[int, ...], Tensor]

    Examples
    --------
    >>> import torch
    >>> def fn(x):
    ...    return 1.0 + x + x**2 + x**3 + x**4 + x**5
    >>> x = torch.tensor([0.0])
    >>> s = series((5, ), fn, x, retain=False)
    >>> derivative(s, (1, ))
    {(0,): tensor([1.]),
     (1,): tensor([2.]),
     (2,): tensor([3.]),
     (3,): tensor([4.]),
     (4,): tensor([5.])}

    """
    index:Tensor = torch.tensor(index)
    keys:Tensor = torch.tensor([*series.keys()])
    values:Tensor = torch.stack([*series.values()])
    output:dict[tuple[int, ...], Tensor] = {}
    for key, value in zip(keys, values):
        if -1 not in torch.sign(key - index):
            factor = key.prod()
            output[tuple((key - index).tolist())] = factor*value
    return output
