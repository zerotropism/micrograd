"""Neurons, layers and multi-layer perceptrons, built on the Value engine."""

import random

from micrograd.engine import Value


class Neuron:
    """Neuron class that takes number of inputs to the neuron 'nin' and computes the output of
    the neuron using the tanh activation function.

    Methods:
        __call__: Computes the output of the neuron using the `tanh` activation function.
        parameters: Returns a list of the parameters of the neuron.
    """

    # constructor takes number of inputs to the neuron 'nin'
    def __init__(self, nin):
        """Constructor for the Neuron class.

        Args:
            nin (int): number of inputs to the neuron.
        """
        # weights
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # bias controlling overall 'trigger happiness' of the neuron
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """Computes the output of the neuron using the `tanh` activation function.\n
        First computes ponderated sum of the inputs and then applies the `tanh` activation function.

        Args:
            x (list): neuron entries.

        Returns:
            Neuron: output of the neuron using the `tanh` activation function.
        """
        # w * x + b
        # raw activation function
        # with b value as the start of the sum instead of default 0.0 for efficiency
        act = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        # to be passed in non-linearity
        out = act.tanh()
        return out

    def parameters(self):
        """Returns the parameters of the neuron.

        Returns:
            list: lists the parameters of the neuron.
        """
        return self.w + [self.b]


class Layer:
    """Layer class that takes number of inputs to the layer 'nin' and number of neurons in a
    single layer 'nout'.

    Methods:
        __call__: Computes the output of the layer by calling each neuron in the layer.
        parameters: Returns a list of the parameters of the layer.
    """

    # 'nin' as number of inputs and 'nout' as number of neurons in a single layer
    def __init__(self, nin, nout):
        """Constructor for the Layer class.

        Args:
            nin (int): number of inputs to the layer.
            nout (int): number of neurons in a single layer.
        """
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """Computes the output of the layer by calling each neuron in the layer.

        Args:
            x (list): layer entries.

        Returns:
            list or int: output of the layer.
        """
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """Returns the parameters of the layer.

        Returns:
            list: lists the parameters of the layer.
        """
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """MLP class that takes number of inputs to the MLP 'nin' and listifying the size of each
    layer in the MLP 'nouts'.

    Methods:
        __call__: Computes the output of the MLP by calling each layer in the MLP.
        parameters: Returns a list of the parameters of the MLP.
    """

    # 'nin' as number of inputs and 'nouts' as number of neurons in each layer listifying the size
    # of each layer in the MLP
    def __init__(self, nin, nouts):
        """Constructor for the MLP class.

        Args:
            nin (int): number of inputs to the MLP.
            nouts (int): number of neurons in each layer.
        """
        # put all together in a list
        sz = [nin] + nouts
        # iterate over consecutive pairs of defined sizes and create Layer objects for them
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    # call them sequentially
    def __call__(self, x):
        """Computes the output of the MLP by calling each layer in the MLP.

        Args:
            x (list): MLP entries.

        Returns:
            list: output of the MLP.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Returns the parameters of the MLP.

        Returns:
            list: lists the parameters of the MLP.
        """
        return [p for layer in self.layers for p in layer.parameters()]
