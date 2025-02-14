import math
import random


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


class Neuron:

    # constructor takes number of inputs to the neuron 'nin'
    def __init__(self, nin):
        # weights
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # bias controlling overall 'trigger happiness' of the neuron
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        # raw activation function
        # with b value as the start of the sum instead of default 0.0 for efficiency
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # to be passed in non-linearity
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:

    # 'nin' as number of inputs and 'nout' as number of neurons in a single layer
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

    # 'nin' as number of inputs and 'nouts' as number of neurons in each layer listifying the size of each layer in the MLP
    def __init__(self, nin, nouts):
        # put all together in a list
        sz = [nin] + nouts
        # iterate over consecutive pairs of defined sizes and create Layer objects for them
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    # call them sequentially
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
