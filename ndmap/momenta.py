"""
Momenta
-------

Compute momenta given coordinates and table representation of mapping

"""

from typing import TypeAlias
from typing import Callable
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from ndmap.evaluate import evaluate
from ndmap.propagate import identity
from ndmap.propagate import propagate
from ndmap.inverse import inverse

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


def momenta(order:tuple[int, ...],
            state:State,
            knobs:Knobs,
            data:Table, *,
            solve:Optional[Callable]=None,
            jacobian:Optional[Callable]=None) -> Table:
    """
    Compute momenta given coordinates and table representation of mapping

    Note, input table is assumed to represent a mapping
    Which is assumed to map (parametric) zero to (parametric) zero
    Input state and knobs are deviations and should equal to zero

    Parameters
    ----------
    order: tuple[int, ...]
        computation order
    state: State
        state
    knobs: Knobs
        knobs value
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
    >>> from ndmap.derivative import derivative
    >>> from ndmap.signature import chop
    >>> from ndmap.evaluate import evaluate
    >>> def quad(x, w, k, l, n=10):
    ...    (qx, px, qy, py), (w, ), (k, ), l = x, w, k, 0.5*l/n
    ...    for _ in range(n):
    ...        qx, qy = qx + l*px/(1 + w), qy + l*py/(1 + w)
    ...        px, py = px - 2.0*l*k*qx, py + 2.0*l*k*qy
    ...        qx, qy = qx + l*px/(1 + w), qy + l*py/(1 + w)
    ...    return torch.stack([qx, px, qy, py])
    >>> x = torch.tensor(4*[0.0], dtype=torch.float64)
    >>> w = torch.tensor(1*[0.0], dtype=torch.float64)
    >>> k = torch.tensor(1*[1.0], dtype=torch.float64)
    >>> n = 1
    >>> t = derivative(n, quad, x, w, k, 0.5)
    >>> m = momenta((n, ), x, [], t)
    >>> chop(m)
    >>> m
    [tensor([0., 0., 0., 0.], dtype=torch.float64),
    tensor([[1.830833306211e+00, 0.000000000000e+00, -2.086282815440e+00, 0.000000000000e+00],
            [0.000000000000e+00, 2.163469209897e+00, 0.000000000000e+00, -1.918651513900e+00],
            [2.086282815440e+00, 0.000000000000e+00, -1.830833306211e+00, 0.000000000000e+00],
            [0.000000000000e+00, 1.918651513900e+00, 0.000000000000e+00, -2.163469209897e+00]],
            dtype=torch.float64)]
    >>> xi = torch.tensor([0.001, 0.0001, -0.001, -0.0001], dtype=torch.float64)
    >>> quad(x + xi, w, k, 0.5)
    tensor([9.254897235990e-04, -3.918654049191e-04, -1.179718773054e-03, -6.336337279403e-04],
        dtype=torch.float64)
    >>> xf = evaluate(t, [xi])
    >>> xf
    tensor([9.254897235990e-04, -3.918654049191e-04, -1.179718773054e-03, -6.336337279403e-04],
        dtype=torch.float64)
    >>> qi, _ = xi.reshape(-1, 2).T
    >>> qf, _ = xf.reshape(-1, 2).T
    >>> qs = torch.cat([qf, qi])
    >>> qs
    tensor([9.254897235990e-04, -1.179718773054e-03, 1.000000000000e-03, -1.000000000000e-03],
        dtype=torch.float64)
    >>> _, pi, = xi.reshape(-1, 2).T
    >>> _, pf, = xf.reshape(-1, 2).T
    >>> ps = torch.cat([pf, pi])
    >>> ps
    tensor([-3.918654049191e-04, -6.336337279403e-04, 1.000000000000e-04, -1.000000000000e-04],
        dtype=torch.float64)
    >>> evaluate(m, [qs])
    tensor([-3.918654049191e-04, -6.336337279403e-04,  1.000000000000e-04,
            -1.000000000000e-04], dtype=torch.float64)

    >>> import torch
    >>> from ndmap.derivative import derivative
    >>> from ndmap.signature import chop
    >>> from ndmap.evaluate import evaluate
    >>> def quad(x, w, k, l, n=10):
    ...    (qx, px, qy, py), (w, ), (k, ), l = x, w, k, 0.5*l/n
    ...    for _ in range(n):
    ...        qx, qy = qx + l*px/(1 + w), qy + l*py/(1 + w)
    ...        px, py = px - 2.0*l*k*qx, py + 2.0*l*k*qy
    ...        qx, qy = qx + l*px/(1 + w), qy + l*py/(1 + w)
    ...    return torch.stack([qx, px, qy, py])
    >>> x = torch.tensor(4*[0.0], dtype=torch.float64)
    >>> w = torch.tensor(1*[0.0], dtype=torch.float64)
    >>> k = torch.tensor(1*[0.0], dtype=torch.float64)
    >>> n = (1, 3, 3)
    >>> t = derivative(n, lambda x, w, k: quad(x, w, 1.0 + k, 0.5), x, w, k)
    >>> m = momenta(n, x, [w, k], t)
    >>> chop(m)
    >>> xi = torch.tensor([0.001, 0.0001, -0.001, -0.0001], dtype=torch.float64)
    >>> dw = torch.tensor(1*[0.001], dtype=torch.float64)
    >>> dk = torch.tensor(1*[0.001], dtype=torch.float64)
    >>> quad(x + xi, w + dw, k + dk, 0.5)
    tensor([ 1.049823087855e-03,  9.948753334896e-05, -1.050077017243e-03, -1.005125083744e-04],
        dtype=torch.float64)
    >>> xf = evaluate(t, [xi, dw, dk])
    >>> xf
    tensor([9.254418393433e-04, -3.923450260824e-04, -1.179666705183e-03, -6.341546017848e-04],
        dtype=torch.float64)
    >>> qi, _ = xi.reshape(-1, 2).T
    >>> qf, _ = xf.reshape(-1, 2).T
    >>> qs = torch.cat([qf, qi])
    >>> qs
    tensor([9.254418393433e-04, -1.179666705183e-03, 1.000000000000e-03, -1.000000000000e-03],
        dtype=torch.float64)
    >>> _, pi, = xi.reshape(-1, 2).T
    >>> _, pf, = xf.reshape(-1, 2).T
    >>> ps = torch.cat([pf, pi])
    >>> ps
    tensor([-3.923450260824e-04, -6.341546017848e-04, 1.000000000000e-04, -1.000000000000e-04],
        dtype=torch.float64)
    >>> evaluate(m, [qs, 0*dw, 0*dk])
    tensor([-3.919530730093e-04, -6.335210807037e-04,  9.990009990025e-05, -9.990009989972e-05],
        dtype=torch.float64)
    >>> evaluate(m, [qs, 1*dw, 1*dk])
    tensor([-3.923450260823e-04, -6.341546017844e-04,  1.000000000001e-04, -9.999999999961e-05],
        dtype=torch.float64)

    Note
    ----
    Input table order is not related to computation order in general, table is treated as exact
    Inverse table is computed, which is memory expensive (similar to direct invariant computation)
    Accuracy is mostly defined by input order (can be higher than input table order)
    And is better for small initial coordinates

    """
    if solve is None:
        def solve(matrix, vector):
            return torch.linalg.lstsq(matrix, vector.unsqueeze(1)).solution.squeeze()

    jacobian = torch.func.jacfwd if jacobian is None else jacobian

    dimension = (len(state), *(len(knob) for knob in knobs))

    def qmap(state, *knobs):
        xi = state
        xf = evaluate(data, [state, *knobs])
        qi, _ = xi.reshape(-1, 2).T
        qf, _ = xf.reshape(-1, 2).T
        return torch.cat([qf, qi])

    def pmap(state, *knobs):
        xi = state
        xf = evaluate(data, [state, *knobs])
        _, pi, = xi.reshape(-1, 2).T
        _, pf, = xf.reshape(-1, 2).T
        return torch.cat([pf, pi])

    tq = identity(order, [state, *knobs], jacobian=jacobian)
    tq = propagate(dimension, order, tq, knobs, qmap, jacobian=jacobian)
    tq = inverse(order, state, knobs, tq, solve=solve, jacobian=jacobian)

    tp = identity(order, [state, *knobs], jacobian=jacobian)
    tp = propagate(dimension, order, tp, knobs, pmap, jacobian=jacobian)

    return propagate(dimension, order, tq, knobs, tp)
