# ndtorch, 2022-2023
## i.a.morozov@inp.nsk.su

Higher order partial derivatives computation with respect to one or several tensor-like variables.
Taylor series function approximation (derivative table and series function representation).
Parametric fixed point computation.

# Derivative

Compute higher order function (partial) derivatives.

```python
>>> from ndtorch.ndtorch import derivative
>>> def fn(x):
...     return 1 + x + x**2 + x**3 + x**4 + x**5
... 
>>> import torch
>>> x = torch.tensor(0.0)
>>> derivative(5, fn, x)
[tensor(1.), tensor(1.), tensor(2.), tensor(6.), tensor(24.), tensor(120.)]
```

```python
>>> from ndtorch.ndtorch import derivative
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
>>> from ndtorch.ndtorch import derivative
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
