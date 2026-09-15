"""A scalar-valued autograd engine: one Value per number, gradients by reverse-mode autodiff."""

import math


class Value:
    """Wraps a single value and keep track of its gradient.\n
    Allows computation of simple operators like addition, substraction, multiplication, division,
    exponentiation & hyperbolic tengent of Value object.
    Computes the gradient of the value with respect to some other value i.e. its childs.
    The gradient is computed using the chain rule of calculus and stored in the 'grad' attribute.

    Methods:
        __repr__: String representation of the Value class.
        __add__: Adds two Value objects.
        __radd__: Default addition of two Value objects when unilateral sense is not respected.
        __neg__: Negates a Value object, allowing substraction operation on Value objects.
        __sub__: Substracts two Value objects.
        __mul__: Multiplies two Value objects.
        __rmul__: Default multiplication of two Value objects when unilateral sense is not
            respected.
        __pow__: Computes the power of a Value object.
        __truediv__: Divides two Value objects.
        tanh: Computes the hyperbolic tangent of a Value object.
        exp: Computes the exponentiation of a Value object.
        backward: Backward pass for the Value object.
    """

    # takes single value that it wraps and keeps track of
    def __init__(self, data: int, _children=(), _op="", label=""):
        """Constructor for the Value class.

        Args:
            data (int): single value that it wraps and keeps track of.
            _children (tuple, optional): child values to compute partial derivation. Defaults to ().
            _op (str, optional): operator sign. Defaults to "".
            label (str, optional): string value of its name. Defaults to "".
        """
        self.data = data
        self.grad = 0.0
        # function to compute chain ruled gradients
        self._backward = lambda: None
        # set for optimization in showing the children of the value for operations
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # prints value.data
    def __repr__(self):
        """String representation of the Value class.

        Returns:
            str: string representation of the Value class.
        """
        return f"Value(data={self.data})"

    # adds self + other
    def __add__(self, other: int):
        """Adds two Value objects.

        Args:
            other (int): integer to be added to the value object.

        Returns:
            Value: result of the addition operation.
        """
        # for convenience in adding non Value objects,
        # if other is not a Value object, convert it to one
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        # compute the gradients in the context of an addition operation
        def _backward():
            # we increment ('+=') so we take into account multiple paths to the same value
            # (multivariate chain rule)
            # which is OK as long as we initialize the gradient to 0.0 at the beginning of
            # the backward pass
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    # default addition when unilateral sense is not respected: other + self
    def __radd__(self, other):
        """Default addition of two Value objects when unilateral sense is not respected.

        Args:
            other (int): integer to be added to the value object.

        Returns:
            Value: result of the addition operation.
        """
        return self + other

    # substracts by negation
    # we first implement a negation function: -self
    def __neg__(self):
        """Negates a Value object, allowing substraction operation on Value objects.

        Returns:
            Value: negated value object.
        """
        return -1.0 * self

    # then we can use the negation function to implement substraction: self - other
    def __sub__(self, other):
        """Substracts two Value objects.

        Args:
            other (int): integer to be substracted from the value object.

        Returns:
            Value: result of the substraction operation.
        """
        return self + (-other)

    # multiplies self * other
    def __mul__(self, other):
        """Multiplies two Value objects.

        Args:
            other (int): integer to be multiplied with the value object.

        Returns:
            Value: result of the multiplication operation.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    # default multiplication when unilateral sense is not respected
    # other * self when self * other cannot be processed
    def __rmul__(self, other):
        """Default multiplication of two Value objects when unilateral sense is not respected.

        Args:
            other (int): integer to be multiplied with the value object.

        Returns:
            Value: result of the multiplication operation.
        """
        return self * other

    # computes division as special case of a more general operation
    # we first need a power function: self**other
    def __pow__(self, other):
        """Computes the power of a Value object.

        Args:
            other (int): integer to be used as the power.

        Returns:
            Value: result of the power operation.
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward

        return out

    # then implement the division: self/other
    def __truediv__(self, other):
        """Divides two Value objects.

        Args:
            other (int): integer to be divided with the value object.

        Returns:
            Value: result of the division operation.
        """
        return self * other**-1

    # computes tanh of a value
    def tanh(self):
        """Computes the hyperbolic tangent of a Value object.

        Returns:
            Value: result of the hyperbolic tangent operation.
        """
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward

        return out

    # computes exponentiation
    def exp(self):
        """Computes the exponentiation of a Value object.

        Returns:
            Value: result of the exponentiation operation.
        """
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad = out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """Backward pass for the Value object."""
        # set a topological sort algorithm
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        # build topological order of the graph
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
