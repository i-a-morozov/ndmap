# ndmap, 2022-2023

<p align="center">
  <img width="100" height="100" src="https://github.com/i-a-morozov/ndmap/blob/main/docs/pics/logo.svg">
</p>

Higher order partial derivatives computation with respect to one or several tensor-like variables.
Taylor series function approximation (derivative table and series function representation).
Parametric fixed point computation.

# Install & build

```
$ pip install git+https://github.com/i-a-morozov/ndmap.git@main
```
or
```
$ pip install ndmap -U
```

# Documentation

[https://i-a-morozov.github.io/ndmap/](https://i-a-morozov.github.io/ndmap/)

# Derivative (composable jacobian)

Compute higher order function (partial) derivatives.

```python
>>> from ndmap.derivative import derivative
>>> def fn(x):
...     return 1 + x + x**2 + x**3 + x**4 + x**5
... 
>>> import torch
>>> x = torch.tensor(0.0)
>>> derivative(5, fn, x)
[tensor(1.), tensor(1.), tensor(2.), tensor(6.), tensor(24.), tensor(120.)]
```

```python
>>> from ndmap.derivative import derivative
>>> def fn(x):
...     x1, x2 = x
...     return x1**2 + x1*x2 + x2**2
... 
>>> import torch
>>> x = torch.tensor([0.0, 0.0])
>>> derivative(2, fn, x, intermediate=False)
tensor([[2., 1.],
        [1., 2.]])
```

```python
>>> from ndmap.derivative import derivative
>>> def fn(x, y):
...     x1, x2 = x
...     return x1**2*(1 + y) + x2**2*(1 - y)
... 
>>> import torch
>>> x = torch.tensor([0.0, 0.0])
>>> y = torch.tensor(0.0)
>>> derivative((2, 1), fn, x, y)
[[tensor(0.), tensor(0.)], [tensor([0., 0.]), tensor([0., 0.])], [tensor([[2., 0.],
        [0., 2.]]), tensor([[ 2.,  0.],
        [ 0., -2.]])]]
```

# Derivative (gradient)

Compute higher order function (partial) derivatives.

```python
>>> from ndmap.gradient import series
>>> def fn(x):
...     return 1 + x + x**2 + x**3 + x**4 + x**5
... 
>>> import torch
>>> x = torch.tensor([0.0])
>>> series((5, ), fn, x, retain=False, series=False)
{(0,): tensor([1.]),
 (1,): tensor([1.]),
 (2,): tensor([2.]),
 (3,): tensor([6.]),
 (4,): tensor([24.]),
 (5,): tensor([120.])}
```

```python
>>> from ndmap.gradient import series
>>> def fn(x):
...     x1, x2 = x
...     return x1**2 + x1*x2 + x2**2
...
>>> import torch
>>> x = torch.tensor([0.0, 0.0])
>>> series((2, ), fn, x, intermediate=False, retain=False, series=False)
{(2, 0): tensor(2.), (1, 1): tensor(1.), (0, 2): tensor(2.)}

```

```python
>>> from ndmap.gradient import series
>>> def fn(x, y):
...     x1, x2 = x
...     y1, = y
...     return x1**2*(1 + y1) + x2**2*(1 - y1)
...
>>> import torch
>>> x = torch.tensor([0.0, 0.0])
>>> y = torch.tensor([0.0])
>>> series((2, 1), fn, x, y, retain=False, series=False)
{(0, 0, 0): tensor(0.),
 (0, 0, 1): tensor(0.),
 (1, 0, 0): tensor(0.),
 (0, 1, 0): tensor(0.),
 (1, 0, 1): tensor(0.),
 (0, 1, 1): tensor(-0.),
 (2, 0, 0): tensor(2.),
 (1, 1, 0): tensor(0.),
 (0, 2, 0): tensor(2.),
 (2, 0, 1): tensor(2.),
 (1, 1, 1): tensor(0.),
 (0, 2, 1): tensor(-2.)}
```

# Desription

```python
>>> import ndmap
>>> ndmap.__about__
```

# Animations

Stable and unstable invariant manifolds approximation

<p align="center">
  <img width="576" height="576" src="https://github.com/i-a-morozov/ndmap/blob/main/docs/pics/manifold.gif">
</p>

Collision of fixed points

<p align="center">
  <img width="576" height="576" src="https://github.com/i-a-morozov/ndmap/blob/main/docs/pics/collision.gif">
</p>

Reduce real part of a hyperbolic fixed point

<p align="center">
  <img width="576" height="576" src="https://github.com/i-a-morozov/ndmap/blob/main/docs/pics/change.gif">
</p>
