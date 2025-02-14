import math
import numpy as np
import matplotlib.pyplot as plt
from graph import draw_dot


class Value:

    # takes single value that it wraps and keeps track of
    def __init__(self, data, _children=(), _op="", label=""):
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
        return f"Value(data={self.data})"

    # adds self + other
    def __add__(self, other):
        # for convenience in adding non Value objects, if other is not a Value object, convert it to one
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        # compute the gradients in the context of an addition operation
        def _backward():
            # we increment ('+=') so we take into account multiple paths to the same value (multivariate chain rule)
            # which is OK as long as we initialize the gradient to 0.0 at the beginning of the backward pass
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    # default addition when unilateral sense is not respected: other + self
    def __radd__(self, other):
        return self + other

    # substracts by negation
    # we first implement a negation function: -self
    def __neg__(self):
        return -1.0 * self

    # then we can use the negation function to implement substraction: self - other
    def __sub__(self, other):
        return self + (-other)

    # multiplies self * other
    def __mul__(self, other):
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
        return self * other

    # computes division as special case of a more general operation
    # we first need a power function: self**other
    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward

        return out

    # then implement the division: self/other
    def __truediv__(self, other):
        return self * other**-1

    # computes tanh of a value
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward

        return out

    # computes exponentiation
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad = out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        # set a topological sort algorithm
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# inputs
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
# weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
# bias
b = Value(6.8813735870195432, label="b")
# values
x1w1 = x1 * w1
x1w1.label = "x1*w1"
x2w2 = x2 * w2
x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b
n.label = "n"
# original output
o = n.tanh()
o.label = "o"
# but we decompose it as the sum of the exponentiations
e = (2 * n).exp()
e.label = "e"
o = (e - 1) / (e + 1)
o.label = "o"
o.backward()
draw_dot(o).render("graph", format="svg", cleanup=True, view=True)
