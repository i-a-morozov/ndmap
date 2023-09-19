"""
Jet
---

Convenience class to work with jets (evaluation point & derivative table)

"""

from __future__ import annotations

from typing import TypeAlias
from typing import Callable
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from ndmap.util import flatten
from ndmap.derivative import derivative
from ndmap.signature import signature
from ndmap.signature import get
from ndmap.signature import set
from ndmap.series import series
from ndmap.evaluate import evaluate
from ndmap.evaluate import table
from ndmap.propagate import identity
from ndmap.propagate import propagate


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


class Jet:
    """
    Convenience class to work with jets (evaluation point & derivative table)

    Returns
    -------
    Jet class instance

    Parameters
    ----------
    dimension: tuple[int, ...], positive
        dimensions
    order: tuple[int, ...], non-negative
        maximum derivative orders
    initialize : bool
        flag to initialize identity derivative table, optional, default=True
    point: Optional[Point]
        evaluation point, default=None
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev
    dtype: torch.dtype, default=torch.float64
        data type
    device: torch.device, default=torch.device('cpu')
        data device

    Attributes
    ----------
    dimension: tuple[int, ...], positive
        dimensions
    order: tuple[int, ...], non-negative
        maximum derivative orders
    initialize : bool
        flag to initialize identity derivative table, optional, default=True
    point: Point
        evaluation point, default=None
    jacobian: Optional[Callable]
        torch.func.jacfwd (default) or torch.func.jacrev
    dtype: torch.dtype, default=torch.float64
        data type
    device: torch.device, default=torch.device('cpu')
        data device
    state: State
        state
    knobs: Knobs
        knobs
    table: Table
        table representation
    series: Series
         series representation
    signature: list[tuple[int, ...]]
        derivative table elements bottom elements signatures
    parametetric: Table
        parametric table

    """
    def __init__(self,
                 dimension:tuple[int, ...],
                 order:tuple[int, ...], *,
                 initialize:bool=True,
                 point:Optional[Point]=None,
                 jacobian:Optional[Callable]=None,
                 dtype:torch.dtype=torch.float64,
                 device:torch.device=torch.device('cpu')) -> None:
        """
        Jet initialization

        Parameters
        ----------
        dimension: tuple[int, ...], positive
            dimensions
        order: tuple[int, ...], non-negative
            maximum derivative orders
        initialize : bool
            flag to initialize identity derivative table, optional, default=True
        point: Optional[Point]
            evaluation point
        jacobian: Optional[Callable]
            torch.func.jacfwd (default) or torch.func.jacrev
        dtype: torch.dtype, default=torch.float64
            data type
        device: torch.device, default=torch.device('cpu')
            data device

        Returns
        -------
        None

        Examples
        --------
        >>> import torch
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> Jet((2, 2), (2, 1), initialize=True, point=[x, k], dtype=torch.float64)
        Jet((2, 2), (2, 1))

        """
        jacobian = torch.func.jacfwd if jacobian is None else jacobian

        self.dimension:tuple[int, ...] = dimension
        self.order:tuple[int, ...] = order
        self.initialize:bool = initialize
        self.point:Point = point
        self.jacobian:Callable = jacobian
        self.dtype:torch.dtype = dtype
        self.device:torch.device = device

        if self.point is None:
            self.point = [torch.zeros(i, dtype=self.dtype, device=self.device) for i in self.dimension]
        else:
            self.point = [tensor.to(self.dtype).to(self.device) for tensor in point]

        state, *knobs = self.point
        self.state:State = state
        self.knobs:Knobs = knobs

        self.table = None
        if self.initialize:
            self.table:Table = identity(self.order, self.point, flag=False, jacobian=self.jacobian)


    def evaluate(self, delta:Delta) -> Tensor:
        """
        Evaluate jet derivative table at a given delta deviation

        Parameters
        ----------
        delta: Delta
            delta deviation

        Returns
        -------
        Tensor

        Examples
        --------
        >>> import torch
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> j = Jet((2, 2), (2, 1), initialize=True, point=[x, k], dtype=torch.float32)
        >>> dx = torch.tensor([1.0, 1.0])
        >>> dk = torch.tensor([1.0, 1.0])
        >>> j.evaluate([dx, dk])
        tensor([1., 1.])

        """
        return evaluate(self.table, delta)


    @property
    def signature(self) -> Signature:
        """
        Compute derivative table elements bottom elements signatures

        Parameters
        ----------
        None

        Returns
        -------
        Signature
            bottom table elements signatures

        Examples
        --------
        >>> import torch
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> j = Jet((2, 2), (2, 1), initialize=True, point=[x, k], dtype=torch.float32)
        >>> j.signature
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

        """
        return signature(self.table, factor=False)


    @property
    def series(self) -> Series:
        """
        Series representation

        Parameters
        ----------
        None

        Returns
        -------
        Series

        Examples
        --------
        >>> import torch
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> j = Jet((2, 2), (2, 1), initialize=True, point=[x, k], dtype=torch.float32)
        >>> j.series
        {(0, 0, 0, 0): tensor([0., 0.]),
        (0, 0, 1, 0): tensor([0., 0.]),
        (0, 0, 0, 1): tensor([0., 0.]),
        (1, 0, 0, 0): tensor([1., 0.]),
        (0, 1, 0, 0): tensor([0., 1.]),
        (1, 0, 1, 0): tensor([0., 0.]),
        (1, 0, 0, 1): tensor([0., 0.]),
        (0, 1, 1, 0): tensor([0., 0.]),
        (0, 1, 0, 1): tensor([0., 0.]),
        (2, 0, 0, 0): tensor([0., 0.]),
        (1, 1, 0, 0): tensor([0., 0.]),
        (0, 2, 0, 0): tensor([0., 0.]),
        (2, 0, 1, 0): tensor([0., 0.]),
        (2, 0, 0, 1): tensor([0., 0.]),
        (1, 1, 1, 0): tensor([0., 0.]),
        (1, 1, 0, 1): tensor([0., 0.]),
        (0, 2, 1, 0): tensor([0., 0.]),
        (0, 2, 0, 1): tensor([0., 0.])}

        """
        return series(self.dimension, self.order, self.table)


    @property
    def parametetric(self) -> Table:
        """
        Get parametric table (first subtable)

        Parameters
        ----------
        None

        Returns
        -------
        Table

        Examples
        --------
        >>> import torch
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> j = Jet((2, 2), (2, 1), initialize=True, point=[x, k], dtype=torch.float32)
        >>> j.parametetric
        [tensor([0., 0.]),
        tensor([[0., 0.],
                [0., 0.]])]

        """
        table, *_ = self.table
        return table


    @parametetric.setter
    def parametetric(self, value:Table) -> None:
        """
        Set parametric table (first subtable)

        Parameters
        ----------
        value: Table
            parametric table

        Returns
        -------
        None

        Examples
        --------
        >>> import torch
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> j = Jet((2, 2), (2, 1), initialize=True, point=[x, k], dtype=torch.float32)
        >>> j.parametetric = 1
        >>> j.parametetric
        1

        """
        set(self.table, (0, ), value)


    @classmethod
    def from_mapping(cls,
                     dimension:tuple[int, ...],
                     order:tuple[int, ...],
                     point:Point,
                     function:Mapping,
                     *args:tuple,
                     jacobian:Optional[Callable]=None,
                     dtype:torch.dtype=torch.float64,
                     device:torch.device=torch.device('cpu')) -> Jet:
        """
        Jet initialization from mapping

        Parameters
        ----------
        dimension: tuple[int, ...], positive
            dimensions
        order: tuple[int, ...], non-negative
            maximum derivative orders
        point: Point
            evaluation point
        function: Mapping
            input function
        *args: tuple
            additional function arguments
        jacobian: Optional[Callable]
            torch.func.jacfwd (default) or torch.func.jacrev
        dtype: torch.dtype, default=torch.float64
            data type
        device: torch.device, default=torch.device('cpu')
            data device

        Returns
        -------
        Jet

        Examples
        --------
        >>> import torch
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> def mapping(x, k):
        ...     x1, x2 = x
        ...     k1, k2 = k
        ...     return torch.stack([x1*k1, x2*k2])
        >>> j = Jet.from_mapping((2, 2), (1, 1), [x, k], mapping, dtype=torch.float32)
        >>> j.table
        [[tensor([0., 0.]),
        tensor([[0., 0.],
                [0., 0.]])],
        [tensor([[0., 0.],
                [0., 0.]]),
        tensor([[[1., 0.],
                [0., 0.]],
        
                [[0., 0.],
                [0., 1.]]])]]

        """
        jacobian = torch.func.jacfwd if jacobian is None else jacobian

        jet = cls(dimension,
                  order,
                  initialize=False,
                  point=point,
                  jacobian=jacobian,
                  dtype=dtype, device=device)

        jet.table:Table = derivative(order,
                                     function,
                                     *point,
                                     *args,
                                     intermediate=True,
                                     jacobian=jacobian)

        return jet


    @classmethod
    def from_table(cls,
                   dimension:tuple[int, ...],
                   order:tuple[int, ...],
                   point:Point,
                   table:Table,
                   jacobian:Optional[Callable]=None,
                   dtype:torch.dtype=torch.float64,
                   device:torch.device=torch.device('cpu')) -> Jet:
        """
        Jet initialization from table

        Parameters
        ----------
        dimension: tuple[int, ...], positive
            dimensions
        order: tuple[int, ...], non-negative
            maximum derivative orders
        point: Point
            evaluation point
        table: Table
            input (derivative) table
        jacobian: Optional[Callable]
            torch.func.jacfwd (default) or torch.func.jacrev
        dtype: torch.dtype, default=torch.float64
            data type
        device: torch.device, default=torch.device('cpu')
            data device

        Returns
        -------
        Jet

        Examples
        --------
        >>> import torch
        >>> from ndmap.derivative import derivative
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> def mapping(x, k):
        ...     x1, x2 = x
        ...     k1, k2 = k
        ...     return torch.stack([x1*k1, x2*k2])
        >>> t = derivative((1, 1), mapping, [x, k])
        >>> j = Jet.from_table((2, 2), (1, 1), [x, k], t, dtype=torch.float32)
        >>> j.table
        [[tensor([0., 0.]),
        tensor([[0., 0.],
                [0., 0.]])],
        [tensor([[0., 0.],
                [0., 0.]]),
        tensor([[[1., 0.],
                [0., 0.]],
        
                [[0., 0.],
                [0., 1.]]])]]

        """
        jacobian = torch.func.jacfwd if jacobian is None else jacobian

        jet = cls(dimension,
                  order,
                  initialize=False,
                  point=point,
                  jacobian=jacobian,
                  dtype=dtype,
                  device=device)

        jet.table:Table = table

        return jet


    @classmethod
    def from_series(cls,
                    dimension:tuple[int, ...],
                    order:tuple[int, ...],
                    point:Point,
                    series:Series,
                    jacobian:Optional[Callable]=None,
                    dtype:torch.dtype=torch.float64,
                    device:torch.device=torch.device('cpu')) -> Jet:
        """
        Jet initialization from series

        Parameters
        ----------
        dimension: tuple[int, ...], positive
            dimensions
        order: tuple[int, ...], non-negative
            maximum derivative orders
        point: list[Tensor]
            evaluation point
        series: Series
            input series
        jacobian: Optional[Callable]
            torch.func.jacfwd (default) or torch.func.jacrev
        dtype: torch.dtype, default=torch.float64
            data type
        device: torch.device, default=torch.device('cpu')
            data device

        Returns
        -------
        Jet

        >>> import torch
        >>> from ndmap.util import curry_apply
        >>> from ndmap.series import series
        >>> from ndmap.propagate import identity
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> def mapping(x, k):
        ...     x1, x2 = x
        ...     k1, k2 = k
        ...     return torch.stack([x1*k1, x2*k2])
        >>> s = [*identity((1, 1), [x, k], flag=True).keys()]
        >>> s = series(s, curry_apply(mapping, (2, 2)), *x, *k)
        >>> j = Jet.from_series((2, 2), (1, 1), [x, k], s)
        >>> j.table
        [[tensor([0., 0.]),
        tensor([[0., 0.],
                [0., 0.]])],
        [tensor([[0., 0.],
                [0., 0.]]),
        tensor([[[1., 0.],
                [0., 0.]],
        
                [[0., 0.],
                [0., 1.]]])]]

        """
        jacobian = torch.func.jacfwd if jacobian is None else jacobian

        jet = cls(dimension,
                  order,
                  initialize=False,
                  point=point,
                  jacobian=jacobian,
                  dtype=dtype,
                  device=device)

        jet.table:Table = table(dimension,
                                order,
                                series,
                                jacobian=jacobian)
        return jet


    def propagate(self,
                  function:Mapping,
                  *pars:tuple) -> Jet:
        """
        Propagate jet.

        Parameters
        ----------
        function: Mapping
            input function
        knobs: Knobs
            input function knobs
        *pars: tuple
            additional function arguments

        Returns
        -------
        Jet

        Examples
        --------
        >>> import torch
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> def mapping(x, k):
        ...     x1, x2 = x
        ...     k1, k2 = k
        ...     return torch.stack([x1*k1, x2*k2])
        >>> j = Jet((2, 2), (1, 1), point=[x, k], dtype=torch.float32)
        >>> j.table
        [[tensor([0., 0.]),
        tensor([[0., 0.],
                [0., 0.]])],
        [tensor([[1., 0.],
                [0., 1.]]),
        tensor([[[0., 0.],
                [0., 0.]],
        
                [[0., 0.],
                [0., 0.]]])]]
        >>> j.propagate(mapping).table
        [[tensor([0., 0.]),
        tensor([[0., 0.],
                [0., 0.]])],
        [tensor([[0., 0.],
                [0., 0.]]),
        tensor([[[1., 0.],
                [0., 0.]],
        
                [[0., 0.],
                [0., 1.]]])]]

        """
        table = propagate(self.dimension,
                          self.order,
                          self.table,
                          self.knobs,
                          function,
                          *pars,
                          intermediate=True,
                          jacobian=self.jacobian)

        return self.from_table(self.dimension,
                               self.order,
                               self.point,
                               table,
                               jacobian=self.jacobian,
                               dtype=self.dtype,
                               device=self.device)


    def compliant(self, other:Jet) -> bool:
        """
        Check jets are compliant (can be composed)

        Parameters
        ----------
        other: Jet
            other jet

        Returns
        -------
        bool

        Examples
        --------
        >>> j1 = Jet((2, 2), (1, 1))
        >>> j2 = Jet((2, 2), (2, 1))
        >>> j3 = Jet((2, 2), (1, 1))
        >>> j1.compliant(j2)
        False
        >>> j1.compliant(j3)
        True

        """
        if not all(i == j for i, j in zip(self.dimension, other.dimension)):
            return False

        if not all(i == j for i, j in zip(self.order, other.order)):
            return False

        return True


    def compose(self, other:Jet) -> Jet:
        """
        Compose jets (evaluate other jet at self jet)

        Parameters
        ----------
        other: Jet
            other jet

        Returns
        -------
        Jet

        Examples
        --------
        >>> j1 = Jet((2, 2), (1, 1), dtype=torch.float32)
        >>> j2 = Jet((2, 2), (1, 1), dtype=torch.float32)
        >>> j1.compose(j2).table
        [[tensor([0., 0.]),
        tensor([[0., 0.],
                [0., 0.]])],
        [tensor([[1., 0.],
                [0., 1.]]),
        tensor([[[0., 0.],
                [0., 0.]],
        
                [[0., 0.],
                [0., 0.]]])]]

        """
        def auxiliary(*args) -> Tensor:
            return other.evaluate([*args])

        return self.propagate(auxiliary)


    def __bool__(self) -> bool:
        """
        Check if table is not None

        Parameters
        ----------
        None

        Returns
        -------
        bool

        Examples
        --------
        >>> j = Jet((2, 2), (1, 1), initialize=False)
        >>> bool(j)
        False
        >>> j = Jet((2, 2), (1, 1), initialize=True)
        >>> bool(j)
        True

        """
        return self.table is not None


    def __eq__(self, other:Jet) -> bool:
        """
        Compare jets

        Parameters
        ----------
        other: Jet
            other jet

        Returns
        -------
        bool

        Examples
        --------
        >>> Jet((2, 2), (1, 1)) == Jet((2, 2), (1, 1))
        True
        >>> Jet((1, 2), (1, 1)) == Jet((2, 2), (1, 1))
        False
        >>> Jet((2, 2), (1, 1)) == Jet((2, 2), (2, 1))
        False

        """
        if not self.compliant(other):
            return False

        if all(torch.allclose(x, y) for x, y in zip(self.series.values(), other.series.values())):
            return True

        return False


    def __getitem__(self, index:tuple[int, ...]) -> Union[Tensor, Table]:
        """
        Get item (derivative table bottom element or subtable)

        Parameters
        ----------
        index: tuple[int, ...]
            index

        Returns
        ----------
        Union[Tensor, Table]

        """
        return get(self.table, index)


    def __setitem__(self, index:tuple[int, ...], value:Union[Tensor, Table]) -> None:
        """
        Set item (derivative table bottom element or subtable)

        Parameters
        ----------
        index: tuple[int, ...]
            index
        value: Union[Tensor, Table]
            value to set

        Returns
        -------
        None

        """
        set(self.table, index, value)


    def __iter__(self):
        """
        Jet iteration (use for unpacking)

        """
        return flatten(self.table, target=list)


    def __len__(self) -> int:
        """
        Jet length (signature length / number of tensors)

        Parameters
        ----------
        None

        Returns
        -------
        int

        """
        return len(self.signature)


    def __call__(self, delta:Delta) -> torch.Tensor:
        """
        Evaluate jet derivative table at a given delta deviation

        Parameters
        ----------
        delta: Delta
            delta deviation

        Returns
        -------
        Tensor

        Examples
        --------
        >>> import torch
        >>> x = torch.tensor([0.0, 0.0])
        >>> k = torch.tensor([0.0, 0.0])
        >>> j = Jet((2, 2), (2, 1), initialize=True, point=[x, k], dtype=torch.float32)
        >>> dx = torch.tensor([1.0, 1.0])
        >>> dk = torch.tensor([1.0, 1.0])
        >>> j.evaluate([dx, dk])
        tensor([1., 1.])

        """
        return self.evaluate(delta)


    def __matmul__(self, other:Jet) -> Jet:
        """
        Compose jets (evaluate other jet at self jet)

        Parameters
        ----------
        other: Jet
            other jet

        Returns
        -------
        Jet

        Examples
        --------
        >>> j1 = Jet((2, 2), (1, 1), dtype=torch.float32)
        >>> j2 = Jet((2, 2), (1, 1), dtype=torch.float32)
        >>> j1.compose(j2).table
        [[tensor([0., 0.]),
        tensor([[0., 0.],
                [0., 0.]])],
        [tensor([[1., 0.],
                [0., 1.]]),
        tensor([[[0., 0.],
                [0., 0.]],
        
                [[0., 0.],
                [0., 0.]]])]]

        """
        return self.compose(other)


    def __repr__(self) -> str:
        """
        String representation.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return f'Jet({self.dimension}, {self.order})'
