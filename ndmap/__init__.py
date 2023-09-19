__about__ = """
ndmap is a collection of tools for computation of higher order derivatives (including all intermediate derivatives) with some applications in nonlinear dynamics (accelerator physics)

Compute derivatives of f(x, y, ...) with respect to several tensor arguments x, y, ... and corresponding derivative orders n, m, ...
Input function f(x, y, ...) is expected to return a tensor or a (nested) list of tensors

Compute derivatives and Taylor approximation of f(x, y, ...) with respect to scalar and/or vector tensor arguments x, y, ... and corresponding derivative orders n, m, ...
Evaluation of Taylor approximation at a given deviation point dx, dy, ...
Input function f(x, y, ...) is expected to return a scalar or vector tensor (not a list)

Derivatives are computed by nesting torch.func jacobian functions (forward or reverse jacobians can be used)
Higher order derivatives with respect to several arguments can be computed using different orders for each argument
Derivatives can be used as a surrogate model for a given function with a vector tensor output (Taylor series approximation and evaluation)

Tools for fixed point computation are avaliable for mappings, including (parametric) derivatives of a selected fixed point
Mapping is a function f(x, y, ...): R^n R^m ... -> R^n that returns a tensor with the same shape as its first argument x (assumed to be a vector)
The first argument is referred as state, other arguments used for computation of derivatives are collectively referred as knobs
State and each knob are assumed to be vector-like tensors
Note, phase space observable is a function f(x, y, ...): R^n R^m ... -> R, e.g. a mapping component

Given a function f(x) with a single tensor argument x that returns a tensor or a (nested) list of tensors
Derivatives with respect to x can be computed upto a given order
By default, all intermediate derivatives are returned, but it is possible to return only the highest order derivative

Derivatives are computed by nesting torch.func jacobian functions (forward or reverse jacobians can be used)
Nesting of jacobian generates redundant computations starting from the second order for functions with non-scalar arguments
Identical partial derivatives are recomputed several times
Hence, computation overhead is exponentially increasing for higher orders

Note, there is no redundant computations in the case of the first order derivatives
Redundancy can be avoided if function argument is treated as separate scalar tensors or vector tensors with one component
f(x, y, ...) -> f(x1, x2, ..., xn, y1, y2, ..., ym, ...)

If an input function returns a tensor, then its derivatives are f(x), Dx f(x), Dxx f(x), ... (value, jacobian, hessian, ...)
Note, function value itself is treated as zeros order derivative
If returned value is a scalar tensor, jacobian is a vector (gradient) and hessian is a matrix
Given the derivatives, the input function Taylor approximation at evaluation point can be constructed
f(x + dx) = 1/0! * f(x) + 1/1! * Dx f(x) @ dx + 1/2! * Dxx f(x) @ dx @ dx + ...
A list t(f, x) = [f(x), Dx f(x), Dxx f(x), ...] is called a derivative table, x is an evaluation point, dx is a deviation from evaluation point x
Derivative order can be deduced from table structure and is equal to table length in the case of a single tensor argument
Note, function to evaluate derivative table is avaliable only for input functions with scalar or vector tensor output
This fuction is also arbitrary differentiable

Table representation is redundant in general, instead the result can be represented by series, collection of monomial coefficients
c(n1, n2, ...) * dx1^n1 * dx2^n2 * ... ~ (n1, n2, ...): c(n1, n2, ...), tuple (n1, n2, ...) is called a monomial index
Series representaion can be computed from a given table representation, table can be also computed from a given series representation
Both table and series representations can be evaluated for a given deviation dx
Both table or series can be used to approximate original function (surrogate model)
Note, input function is assumed to return a vector tensor for series representation

For a function with several tensor arguments f(x, y, z, ...), derivatives can be computed separately with respect to each argument
Derivative orders and arguments shapes can be different
Derivative table has a nested structure in the case with several tensor arguments
Combined monomial x1^n1 * x2^n2 * ... * y1^m1 * y2^m2 * ... * z1^k1 * z2^k2 * ... is used in series representation
Again, for series representation, input function is assumed to return a vector
Note, the ordering of derivatives is Dx...xy...yz...z = (Dx...x (Dy...y (Dz...z f)))
Derivatives are computed starting from the last argument
Derivative table structure for f(x), f(x, y) and f(x, y, z) is shown below
Similar structure holds for the case with more arguments

f(x)
t(f, x)
[f, Dx f, Dxx f, ...]

f(x, y)
t(f, x, y)
[
    [    f,     Dy f,     Dyy f, ...],
    [ Dx f,  Dx Dy f,  Dx Dyy f, ...],
    [Dxx f, Dxx Dy f, Dxx Dyy f, ...],
    ...
]

f(x, y, z)
t(f, x, y, z)
[
    [
        [         f,          Dz f,          Dzz f, ...],
        [      Dy f,       Dy Dz f,       Dy Dzz f, ...],
        [     Dyy f,      Dyy Dz f,      Dyy Dzz f, ...],
        ...
    ],
    [
        [      Dx f,       Dx Dz f,       Dx Dzz f, ...],
        [   Dx Dy f,    Dx Dy Dz f,    Dx Dy Dzz f, ...],
        [  Dx Dyy f,   Dx Dyy Dz f,   Dx Dyy Dzz f, ...],
        ...
    ],
    [
        [    Dxx f,     Dxx Dz f,     Dxx Dzz f, ...],
        [ Dxx Dy f,  Dxx Dy Dz f,  Dxx Dy Dzz f, ...],
        [Dxx Dyy f, Dxx Dyy Dz f, Dxx Dyy Dzz f, ...],
        ...
    ],
    ...
]

Each tensor at the bottom level of a derivative table is associated with a signature, e.g. Dxxyzz f = Dxx Dy Dzz f ~ (2, 1, 2)
Signature is a tuple of corresponding orders (not the same as monomial index)
Bottom table elements signatures can be computed with signature function
Given a bottom element signature, it can be extracted or assigned a new value
Note, subtables can also be extracted and modified

Bottom table elements are related to corresponding monomials
Given a signature (n, m, ...), possible monomial indices are (n1, n2, ..., m1, m2, ...) with n1 + n2 + ... = n, m1 + m2 + ... = m, ...
Monomial index table with repetitions can be generated with index function
Note, index table corresponds to reversed bottom table element if the number of function arguments is greater than one

Mapping is a function f(x, y, z, ...) that returns a tensor with the same shape as its first argument x (assumed to be a vector)
For mappings, x corresponds to dynamical variable (state), while other variables are parameters (knobs)
Composition of mappings f and g is also a mapping h = g(f(x, y, z, ...), y, z, ...)
Derivatives of h can be computed directly or by propagation of t(f, x, y, z, ...), derivatives of f(x, y, z, ...), thought g(x, y, z, ...)
This can be done with propagate function which takes table (or series) as a parameter
Dynamical variable x can be represented as identity table (or series), can be generated with identity function
Note, for composition/propagation functions are expected to have identical signatures (state, *knobs, ...), fixed parameters can be different

Given a mapping h = g(f(x, y, z, ...), y, z, ...), table t(h, x, y, z, ...) is equivalent to composition of tables t(f, x, y, z, ...) and t(g, x, y, z, ...)
If mappings f and g have no constant terms (map zero to zero)
This is a homomorphism property of truncated polynomials composition without constant terms

Given a mapping f(x, y, z, ...), its (dynamical) fixed points of given period can be computed, x(n) := f^n(x(n), y, z, ...)
Note, such fixed points might not exist for a general mapping
Fixed points are computed with newton method as roots of F = x(n) - f^n(x(n), y, z, ...) = 0

Fixed points can depend of parameters/knobs (y, z, ...)
Parametric fixed point can computed for given dynamical fixed point, i.e. derivatives of a fixed point with respect to parameters

Jet class provides utilities for handling derivivative tables (generation, propagation, composition and other)
Here jet is an evaluation point and corresponding derivative table (also variable dimensions, orders and othe parameters)

Other functionality relevant to applications in nonlinear dynamics is planned to be added.

"""