"""
Coupled twiss parameters computation.

Ivan Morozov, 2022-2023

"""
from __future__ import annotations

import numpy
import torch

from torch import Tensor

def mod(x:float | Tensor,
        y:float | Tensor,
        z:float | Tensor=0.0) -> float | Tensor:
    """
    Return the remainder on division of x by y with offset z.

    Note, float value is returned

    """
    return x - ((x - z) - (x - z) % y)


def block_symplectic(*,
                     dtype:torch.dtype=torch.float64,
                     device:torch.device=torch.device('cpu')) -> Tensor:
    """
    Generate 2D symplectic block [[0, 1], [-1, 0]] matrix.

    Parameters
    ----------
    dtype: torch.dtype, default=torch.float64
        data type
    device: torch.device, torch.device=torch.device('cpu')
        data device

    Returns
    -------
    symplectic block (Tensor)

    """
    return torch.tensor([[0, 1], [-1, 0]], dtype=dtype, device=device)


def block_rotation(alpha:Tensor) -> Tensor:
    """
    Generate 2D rotation block [[cos(alpha), sin(alpha)], [-sin(alpha), cos(alpha)]] matrix.

    Parameters
    ----------
    alpha: Tensor
        rotation angle

    Returns
    -------
    rotation block (Tensor)

    """
    return torch.stack(
        [
            torch.stack([+alpha.cos(), +alpha.sin()]),
            torch.stack([-alpha.sin(), +alpha.cos()])
        ]
    )


def id_symplectic(d:int,
                  *,
                  dtype:torch.dtype=torch.float64,
                  device:torch.device=torch.device('cpu')) -> Tensor:
    """
    Generate symplectic identity matrix for a given (configuration space) dimension.

    Note, symplectic block is [[0, 1], [-1, 0]], i.e. canonical variables are grouped by pairs

    Parameters
    ----------
    d: int, positive
        configuration space dimension
    dtype: torch.dtype, default=torch.float64
        data type
    device: torch.device, torch.device=torch.device('cpu')
        data device

    Returns
    -------
    symplectic identity matrix (torch.Tensor)

    """
    block = block_symplectic(dtype=dtype, device=device)
    return torch.block_diag(*[block for _ in range(d)])


def is_symplectic(m:Tensor,
                  *,
                  epsilon:float=1.0E-12) -> Tensor:
    """
    Test symplectic condition for a given input matrix elementwise.

    Can be mapped over a batch of matrices
    (k, 2*n, 2*n) -> (k)

    Parameters
    ----------
    m: Tensor, even-dimension
        input matrix to test
    epsilon: float, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    test result (Tensor)

    """
    d = len(m) // 2
    s = id_symplectic(d, dtype=m.dtype, device=m.device)
    return epsilon > (m.T @ s @ m - s).abs().max()


def to_symplectic(m:Tensor) -> Tensor:
    """
    Perform symplectification of a given input matrix.

    Can be mapped over a batch of matrices
    (k, 2*n, 2*n) -> (k)

    Parameters
    ----------
    m: Tensor, even-dimension
        input matrix to symplectify

    Returns
    -------
    symplectified matrix (Tensor)

    """
    d = len(m) // 2
    i = torch.eye(2*d, dtype=m.dtype, device=m.device)
    s = id_symplectic(d, dtype=m.dtype, device=m.device)
    v = s @ (i - m) @ (i + m).inverse()
    w = 0.5*(v + v.T)
    return (s + w).inverse() @ (s - w)


def inverse_symplectic(m:Tensor) -> Tensor:
    """
    Compute inverse of a given input symplectic matrix.

    Can be mapped over a batch of matrices
    (k, 2*n, 2*n) -> (k)

    Parameters
    ----------
    m: Tensor, even-dimension, symplectic
        input matrix

    Returns
    -------
    inverse matrix (Tensor)

    """
    d = len(m) // 2
    s = id_symplectic(d, dtype=m.dtype, device=m.device)
    return -s @ m.T @ s


def rotation(*angle:tuple[Tensor, ...]) -> Tensor:
    """
    Generate rotation matrix using given angles.

    Note, block rotation is [[cos(alpha), sin(alpha)], [-sin(alpha), cos(alpha)]]

    Can be mapped over a batch of angles
    (k, n, ...) -> (k, 2*n, 2*n)

    Parameters
    ----------
    angle: tuple[Tensor, ...]
        rotation angles

    Returns
    -------
    rotation matrix (Tensor)

    """
    return torch.block_diag(*torch.func.vmap(block_rotation)(torch.stack(angle)))


def check(m:Tensor,
          *,
          epsilon:float=1.0E-12) -> Tensor:
    """
    Check one-turn matrix stability.

    Can be mapped over a batch of matrices
    (k, 2*n, 2*n) -> (k)

    Parameters
    ----------
    m: Tensor, even-dimension
        input one-turn matrix
    epsilon: float, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    check result (Tensor)

    """
    l = torch.linalg.eigvals(m).log()
    return ((l.real.abs() < epsilon)*(l.imag.abs() > epsilon)).sum() == len(l)


def twiss(m:Tensor,
          *,
          epsilon:float=1.0E-12) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute fractional tunes, normalization matrix and Wolski twiss matrices for a given one-turn input matrix.

    Note, matrix stability is not checked

    Input matrix can have arbitrary even dimension
    In-plane 'beta' values are used for ordering

    Symplectic block is [[0, 1], [-1, 0]], i.e. canonical variables are grouped by pairs
    Rotation block is [[cos(alpha), sin(alpha)], [-sin(alpha), cos(alpha)]]
    Complex block is 1/sqrt(2)*[[1, 1j], [1, -1j]]

    Can be mapped over a batch of matrices
    (k, 2*n, 2*n) -> ((k, n), (k, 2*n, 2*n), (k, n, 2*n, 2*n))

    Parameters
    ----------
    m: Tensor, even-dimension, symplectic, stable
        input (stable) one-turn matrix
    epsilon: float, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    fractional tunes [T_1, ..., T_K], normalization matrix N and Wolski twiss matrices W = [W_1, ..., W_K] (tuple[Tensor, Tensor, Tensor])
    M = N R N^-1 = ... + W_I S sin(2*pi*T_I) - (W_I S)**2 cos(2*pi*T_I) + ... for I = 1, ..., K

    """
    dtype = m.dtype
    device = m.device

    rdtype = torch.tensor(1, dtype=dtype).abs().dtype
    cdtype = (1j*torch.tensor(1, dtype=dtype)).dtype

    d = len(m) // 2

    b_p = torch.tensor([[1, 0], [0, 1]], dtype=rdtype, device=device)
    b_s = torch.tensor([[0, 1], [-1, 0]], dtype=rdtype, device=device)
    b_c = 0.5**0.5*torch.tensor([[1, +1j], [1, -1j]], dtype=cdtype, device=device)

    m_p = torch.stack([torch.block_diag(*[b_p*(i == j) for i in range(d)]) for j in range(d)])
    m_s = torch.block_diag(*[b_s for _ in range(d)])
    m_c = torch.block_diag(*[b_c for _ in range(d)])

    l, v = torch.linalg.eig(m)
    l, v = l.reshape(d, -1), v.T.reshape(d, -1, 2*d)
    for i, (v1, v2) in enumerate(v):
        v[i] /= (-1j*(v1 @ m_s.to(cdtype) @ v2)).abs().sqrt()

    for i in range(d):
        o = torch.clone(l[i].log()).imag.argsort()
        l[i], v[i] = l[i, o], v[i, o]

    t = 1.0 - l.log().abs().mean(-1)/(2.0*numpy.pi)

    n = torch.cat([*v]).H
    n = (n @ m_c).real
    w = n @ m_p @ n.T

    o = torch.stack([w[i].diag().argmax() for i in range(d)]).argsort()
    t, v = t[o], v[o]
    n = torch.cat([*v]).H
    n = (n @ m_c).real

    f = (torch.stack(torch.hsplit(n.T @ m_s @ n - m_s, d)).abs().sum((1, -1)) < epsilon)
    g = f.logical_not()
    for i in range(d):
        t[i] = f[i]*t[i] + g[i]*(1.0 - t[i]).abs()
        v[i] = f[i]*v[i] + g[i]*v[i].conj()

    n = torch.cat([*v]).H
    n = (n @ m_c).real

    i = torch.arange(d, dtype=torch.int64, device=device)
    a = (n[2*i, 2*i + 1] + 1j*n[2*i, 2*i]).angle() - 0.5*numpy.pi
    n = n @ rotation(*a)

    w = n @ m_p @ n.T

    return t, n, w


def propagate(twiss:Tensor,
              transport:Tensor) -> Tensor:
    """
    Propagate Wolski twiss matrices throught a given transport matrix.

    Can be mapped over a batch of twiss matrices and transport matrices
    ((k, n, 2*n, 2*n), (k, 2*n, 2*n)) -> (k, n, 2*n, 2*n)

    Parameters
    ----------
    twiss: Tensor, even-dimension
        Wolski twiss matrices
    transport: Tensor, even-dimension, symplectic
        transport matrix

    Returns
    -------
    propagated Wolski twiss matrices (Tensor)

    """
    return transport @ twiss @ transport.T


def advance(normal:Tensor,
            transport:Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute phase advance and final normalization matrix for a given normalization matrix and a given transport matrix.

    Note, output phase advance is mod 2 pi

    Can be mapped over a batch of normalization matrices and transport matrices
    ((k, 2*n, 2*n), (k, 2*n, 2*n)) -> ((k, n), (k, 2*n, 2*n))

    Parameters
    ----------
    normal: Tensor, even-dimension, symplectic
        normalization matrix
    transport: Tensor, even-dimension, symplectic
        transport matrix

    Returns
    -------
    phase advance and final normalization matrix (tuple[Tensor, Tensor])

    """
    index = torch.arange(len(normal) // 2, dtype=torch.int64, device=normal.device)
    local = transport @ normal
    alpha = mod(torch.arctan2(local[2*index, 2*index + 1], local[2*index, 2*index]), 2.0*numpy.pi)
    return alpha, local @ rotation(*(-alpha))


def normal_to_wolski(normal:Tensor) -> Tensor:
    """
    Compute Wolski twiss matrices for a given normalization matrix.

    Can be mapped over a batch of normalization matrices
    (k, 2*n, 2*n) -> (k, n, 2*n, 2*n)

    Parameters
    ----------
    normal: torch.Tensor, even-dimension, symplectic
        normalization matrix

    Returns
    -------
    Wolski twiss matrices (Tensor)

    """
    d = len(normal) // 2
    p = torch.tensor([[1, 0], [0, 1]], dtype=normal.dtype, device=normal.device)
    p = torch.stack([torch.block_diag(*[p*(i == j) for i in range(d)]) for j in range(d)])
    return normal @ p @ normal.T


def wolski_to_normal(wolski:Tensor,
                     *,
                     epsilon:float=1.0E-12) -> Tensor:
    """
    Compute normalization matrix for given Wolski twiss matrices.

    Note, normalization matrix is computed using fake one-turn matrix with fixed tunes

    Can be mapped over a batch of Wolski twiss matrices
    (k, n, 2*n, 2*n) -> (k, 2*n, 2*n)

    Parameters
    ----------
    wolski: Tensor
        Wolski twiss matrices
    epsilon: float, optional, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    normalization matrix (Tensor)

    """
    d = len(wolski)
    block = torch.tensor([[0, 1], [-1, 0]], dtype=wolski.dtype, device=wolski.device)
    block = torch.block_diag(*[block for _ in range(d)])

    tune = 2.0**0.5*torch.linspace(1, d, d, dtype=wolski.dtype, device=wolski.device)

    data = wolski[0] @ block
    turn = data*tune[0].sin() - data @ data*tune[0].cos()
    for i in range(1, d):
        data = wolski[i] @ block
        turn += data*tune[i].sin() - data @ data*tune[i].cos()

    _, normal, _ = twiss(turn, epsilon=epsilon)

    return normal


def parametric_normal(twiss:Tensor) -> Tensor:
    """
    Generate 'parametric' 4x4 normalization matrix for given free elements.

    Note, elements denoted with X are computed for given free elements using symplectic condition constraints
    For n11 > 0 & n33 > 0, all matrix elements are not singular in uncoupled limit

        n11   0 n13 n14
        n21   X   X   X
        n31   X n33   0
        n41   X n43   X

    Can be mapped over a batch of input free elements
    (k, m) -> (k, 2*n, 2*n)

    Parameters
    ----------
    twiss: Tensor
        free matrix elements
        n11, n33, n21, n43, n13, n31, n14, n41

    Returns
    -------
    normalization matrix (Tensor)
    M = N R N^-1

    """
    n11, n33, n21, n43, n13, n31, n14, n41 = twiss

    i = torch.tensor(1.0, dtype=twiss.dtype, device=twiss.device)
    o = torch.tensor(0.0, dtype=twiss.dtype, device=twiss.device)

    return torch.stack([
        torch.stack([n11, o, n13, n14]),
        torch.stack([n21, n33*(n11 + n14*(n33*n41 - n31*n43))/(n11*(n11*n33 - n13*n31)), (n13*n21 + n33*n41 - n31*n43)/n11, (n14*n21*n33 -n31 + n14*n31/n11*(n31*n43 - n13*n21 - n33*n41))/(n11*n33 - n13*n31)]),
        torch.stack([n31, n14*n33/n11, n33, o]),
        torch.stack([n41, (n13*(-1.0 - (n14*n33*n41)/n11) + n14*n33*n43)/(n11*n33 - n13*n31), n43, (n11 + n14*(n33*n41 - n31*n43))/(n11*n33 - n13*n31)])
    ])


def lb_normal(twiss:Tensor,
              *,
              epsilon:float=1.0E-12) -> Tensor:
    """
    Generate Lebedev-Bogacz normalization matrix.

    [a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2]
    [a1x, b1x, a2y, b2y] are 'in-plane' twiss parameters

    Can be mapped over a batch of input twiss parameters
    (k, m) -> (k, 2*n, 2*n)

    Parameters
    ----------
    twiss : Tensor
        Lebedev-Bogacz twiss parameters
        a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2
    epsilon: float, optional, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    normalization matrix (Tensor)
    M = N R N^-1

    """
    a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2 = twiss

    i = torch.tensor(1.0, dtype=twiss.dtype, device=twiss.device)
    o = torch.tensor(0.0, dtype=twiss.dtype, device=twiss.device)

    cv1, sv1 = v1.cos(), v1.sin()
    cv2, sv2 = v2.cos(), v2.sin()

    b1x *= b1x > epsilon
    b2x *= b2x > epsilon
    b1y *= b1y > epsilon
    b2y *= b2y > epsilon

    return torch.stack([
        torch.stack([b1x.sqrt(), o, b2x.sqrt()*cv2, -b2x.sqrt()*sv2]),
        torch.stack([-a1x/b1x.sqrt(), (1 - u)/b1x.sqrt(), (-a2x*cv2 + u*sv2)/b2x.sqrt(), (a2x*sv2 + u*cv2)/b2x.sqrt()]),
        torch.stack([b1y.sqrt()*cv1, -b1y.sqrt()*sv1, b2y.sqrt(), o]),
        torch.stack([(-a1y*cv1 + u*sv1)/b1y.sqrt(), (a1y*sv1 + u*cv1)/b1y.sqrt(), -a2y/b2y.sqrt(), (1 - u)/b2y.sqrt()])
    ]).nan_to_num(posinf=0.0, neginf=0.0)


def cs_normal(twiss:Tensor,
              *,
              epsilon:float=1.0E-12) -> Tensor:
    """
    Generate Courant-Snyder normalization matrix.

    [ax, bx, ay, by]

    Can be mapped over a batch of input twiss parameters
    (k, m) -> (k, 2*n, 2*n)

    Parameters
    ----------
    twiss : Tensor
        Courant-Snyder twiss parameters
        ax, bx, ay, by
    epsilon: float, optional, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    normalization matrix (Tensor)
    M = N R N^-1

    """
    ax, bx, ay, by = twiss

    i = torch.tensor(1.0, dtype=twiss.dtype, device=twiss.device)
    o = torch.tensor(0.0, dtype=twiss.dtype, device=twiss.device)

    bx *= bx > epsilon
    by *= by > epsilon

    return torch.stack([
        torch.stack([bx.sqrt(), o, o, o]),
        torch.stack([-ax/bx.sqrt(), 1/bx.sqrt(), o, o]),
        torch.stack([o, o, by.sqrt(), o]),
        torch.stack([o, o, -ay/by.sqrt(), 1/by.sqrt()]),
    ]).nan_to_num(posinf=0.0, neginf=0.0)


def wolski_to_lb(twiss:Tensor) -> Tensor:
    """
    Convert Wolski twiss matrices to Lebedev-Bogacz twiss parameters.

    [a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2]
    [a1x, b1x, a2y, b2y] are 'in-plane' twiss parameters

    Can be mapped over a batch of input twiss parameters
    (k, n, 2*n, 2*n) -> (k, m)

    Parameters
    ----------
    twiss: Tensor
        Wolski twiss matrices

    Returns
    -------
    Lebedev-Bogacz twiss (Tensor)

    """
    a1x = -twiss[0, 0, 1]
    b1x = +twiss[0, 0, 0]
    a2x = -twiss[1, 0, 1]
    b2x = +twiss[1, 0, 0]

    a1y = -twiss[0, 2, 3]
    b1y = +twiss[0, 2, 2]
    a2y = -twiss[1, 2, 3]
    b2y = +twiss[1, 2, 2]

    u = 1/2*(1 + a1x**2 - a1y**2 - b1x*twiss[0, 1, 1] + b1y*twiss[0, 3, 3])

    cv1 = (1/torch.sqrt(b1x*b1y)*twiss[0, 0, 2]).nan_to_num(nan=-1.0)
    sv1 = (1/u*(a1y*cv1 + 1/torch.sqrt(b1x)*(torch.sqrt(b1y)*twiss[0, 0, 3]))).nan_to_num(nan=0.0)

    cv2 = (1/torch.sqrt(b2x*b2y)*twiss[1, 0, 2]).nan_to_num(nan=+1.0)
    sv2 = (1/u*(a2x*cv2 + 1/torch.sqrt(b2y)*(torch.sqrt(b2x)*twiss[1, 1, 2]))).nan_to_num(nan=0.0)

    v1 = torch.arctan2(sv1, cv1)
    v2 = torch.arctan2(sv2, cv2)

    return torch.stack([a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2])


def lb_to_wolski(twiss:Tensor,
                 *,
                 epsilon:float=1.0E-12) -> Tensor:
    """
    Convert Lebedev-Bogacz twiss parameters to Wolski twiss matrices.

    Can be mapped over a batch of input twiss parameters
    (k, m) -> (k, n, 2*n, 2*n)

    Parameters
    ----------
    twiss: Tensor
        Lebedev-Bogacz twiss parameters
        a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2
    epsilon: float, optional, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    Wolski twiss matrices (Tensor)

    """
    normal = lb_normal(twiss, epsilon=epsilon)

    p1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=normal.dtype, device=normal.device)
    p2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=normal.dtype, device=normal.device)

    w1 = normal @ p1 @ normal.T
    w2 = normal @ p2 @ normal.T

    return torch.stack([w1, w2])


def wolski_to_cs(twiss:Tensor) -> Tensor:
    """
    Convert Wolski twiss matrices to Courant-Snyder twiss parameters.

    [ax, bx, ay, by]

    Can be mapped over a batch of input twiss parameters
    (k, n, 2*n, 2*n) -> (k, m)

    Parameters
    ----------
    twiss: Tensor
        Wolski twiss matrices

    Returns
    -------
    Courant-Snyder twiss (Tensor)

    """
    ax = -twiss[0, 0, 1]
    bx = +twiss[0, 0, 0]

    ay = -twiss[1, 2, 3]
    by = +twiss[1, 2, 2]

    return torch.stack([ax, bx, ay, by])


def cs_to_wolski(twiss:Tensor,
                 *,
                 epsilon:float=1.0E-12) -> Tensor:
    """
    Convert Courant-Snyder twiss parameters to Wolski twiss matrices.

    Can be mapped over a batch of input twiss parameters
    (k, m) -> (k, n, 2*n, 2*n)

    Parameters
    ----------
    twiss: Tensor
        Courant-Snyder twiss parameters
        ax, bx, ay, by
    epsilon: float, optional, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    Wolski twiss matrices (Tensor)

    """
    normal = cs_normal(twiss, epsilon=epsilon)

    p1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=normal.dtype, device=normal.device)
    p2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=normal.dtype, device=normal.device)

    w1 = normal @ p1 @ normal.T
    w2 = normal @ p2 @ normal.T

    return torch.stack([w1, w2])


def transport(n1:Tensor,
              n2:Tensor,
              *mu12:tuple[Tensor, ...]) -> Tensor:
    """
    Generate transport matrix using normalization matrices and phase advances between given locations.

    Can be mapped over a batch of input parameters
    ((k, 2*n, 2*n), (k, 2*n, 2*n), k, k, ...) -> (k, 2*n, 2*n)

    Parameters
    ----------
    n1: Tensor, even-dimension, symplectic
        normalization matrix at the 1st location
    n2: Tensor, even-dimension, symplectic
        normalization matrix at the 2nd location
    *mu12: tuple[Tensor, ...]
        phase advances between locations

    Returns
    -------
    transport matrix (Tensor)

    """
    return n2 @ rotation(*mu12) @ n1.inverse()


def lb_transport(twiss1:Tensor,
                 twiss2:Tensor,
                 *mu12:tuple[Tensor, ...],
                 epsilon:float=1.0E-12) -> Tensor:
    """
    Generate transport matrix using LB twiss data between given locations.

    Can be mapped over a batch of input parameters

    Parameters
    ----------
    twiss1: Tensor
        twiss parameters at the 1st location
    twiss2: Tensor
        twiss parameters at the 2nd location
    *mu12: tuple[Tensor, ...]
        phase advances between locations
    epsilon: float, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    transport matrix (Tensor)

    """
    n1 = lb_normal(twiss1)
    n2 = lb_normal(twiss2)
    return transport(n1, n2, *mu12)


def cs_transport(twiss1:Tensor,
                 twiss2:Tensor,
                 *mu12:tuple[Tensor, ...],
                 epsilon:float=1.0E-12) -> Tensor:
    """
    Generate transport matrix using CS twiss data between given locations.

    Can be mapped over a batch of input parameters

    Parameters
    ----------
    twiss1: Tensor
        twiss parameters at the 1st location
    twiss2: Tensor
        twiss parameters at the 2nd location
    *mu12: tuple[Tensor, ...]
        phase advances between locations
    epsilon: float, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    transport matrix (Tensor)

    """
    n1 = cs_normal(twiss1)
    n2 = cs_normal(twiss2)
    return transport(n1, n2, mu12)


def invariant(normal:Tensor,
              orbit:Tensor) -> Tensor:
    """
    Compute linear invariants for a given normalization matrix and orbit.

    Note, orbit is assumed to have the form [..., [qx_i, px_i, qy_i, py_i], ...]

    Can be mapped over a batch of input parameters

    Parameters
    ----------
    normal: Tensor, even-dimension, symplectic
        normalization matrix
    orbit: Tensor
        orbit

    Returns
    -------
    [jx, jy] for each turn (Tensor)

    """
    qx, px, qy, py = torch.inner(inverse_symplectic(normal), orbit)

    jx = 1/2*(qx**2 + px**2)
    jy = 1/2*(qy**2 + py**2)

    return torch.stack([jx, jy])


def lb_invariant(twiss:Tensor,
                 orbit:Tensor,
                 *,
                 epsilon:float=1.0E-12) -> Tensor:
    """
    Compute linear invariants for a given LB twiss and orbit.

    Note, orbit is assumed to have the form [..., [qx_i, px_i, qy_i, py_i], ...]

    Can be mapped over a batch of input parameters

    Parameters
    ----------
    twiss: Tensor
        LB twiss
        a1x, b1x, a2x, b2x, a1y, b1y, a2y, b2y, u, v1, v2
    orbit: Tensor
        orbit
    epsilon: float, optional, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    [jx, jy] for each turn (torch.Tensor)

    """
    normal = lb_normal(twiss, epsilon=epsilon)
    return invariant(normal, orbit)


def cs_invariant(twiss:Tensor,
                 orbit:Tensor,
                 *,
                 epsilon:float=1.0E-12) -> Tensor:
    """
    Compute linear invariants for a given CS twiss and orbit.

    Note, orbit is assumed to have the form [..., [qx_i, px_i, qy_i, py_i], ...]

    Can be mapped over a batch of input parameters

    Parameters
    ----------
    twiss: Tensor
        CS twiss
        ax, bx, ay, by
    orbit: Tensor
        orbit
    epsilon: float, optional, default=1.0E-12
        tolerance epsilon

    Returns
    -------
    [jx, jy] for each turn (Tensor)

    """
    normal = cs_normal(twiss, epsilon=epsilon)
    return invariant(normal, orbit)


def momenta(matrix:Tensor,
            qx1:Tensor,
            qx2:Tensor,
            qy1:Tensor,
            qy2:Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute momenta at positions 1 & 2 for given transport matrix and coordinates at 1 & 2.

    Can be mapped over a batch of input parameters

    Parameters
    ----------
    matrix: torch.Tensor, even-dimension, symplectic
        transport matrix between
    qx1, qx2, qy1, qy2: Tensor
        x & y coordinates at 1 & 2

    Returns
    -------
    px and py at 1 & 2 (Tensor)

    """
    forward = matrix
    inverse = matrix.inverse()

    m11, m12, m13, m14 = forward[0]
    m21, m22, m23, m24 = forward[1]
    m31, m32, m33, m34 = forward[2]
    m41, m42, m43, m44 = forward[3]

    px1  = qx1*(m11*m34 - m14*m31)/(m14*m32 - m12*m34)
    px1 += qx2*m34/(m12*m34 - m14*m32)
    px1 += qy1*(m13*m34 - m14*m33)/(m14*m32 - m12*m34)
    px1 += qy2*m14/(m14*m32 - m12*m34)

    py1  = qx1*(m11*m32 - m12*m31)/(m12*m34 - m14*m32)
    py1 += qx2*m32/(m14*m32 - m12*m34)
    py1 += qy1*(m12*m33 - m13*m32)/(m14*m32 - m12*m34)
    py1 += qy2*m12/(m12*m34 - m14*m32)

    m11, m12, m13, m14 = inverse[0]
    m21, m22, m23, m24 = inverse[1]
    m31, m32, m33, m34 = inverse[2]
    m41, m42, m43, m44 = inverse[3]

    px2  = qx2*(m11*m34 - m14*m31)/(m14*m32 - m12*m34)
    px2 += qx1*m34/(m12*m34 - m14*m32)
    px2 += qy2*(m13*m34 - m14*m33)/(m14*m32 - m12*m34)
    px2 += qy1*m14/(m14*m32 - m12*m34)

    py2  = qx2*(m11*m32 - m12*m31)/(m12*m34 - m14*m32)
    py2 += qx1*m32/(m14*m32 - m12*m34)
    py2 += qy2*(m12*m33 - m13*m32)/(m14*m32 - m12*m34)
    py2 += qy1*m12/(m12*m34 - m14*m32)

    return torch.stack([px1, py1]), torch.stack([px2, py2])

def main():
    pass

if __name__ == '__main__':
    main()
