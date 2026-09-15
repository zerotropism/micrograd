"""The network layer: shapes, parameter counts, and that training actually descends."""

import random

from micrograd.engine import Value
from micrograd.nn import MLP, Layer, Neuron


def test_a_neuron_has_one_weight_per_input_plus_a_bias() -> None:
    assert len(Neuron(3).parameters()) == 4


def test_a_layer_holds_one_neuron_per_output() -> None:
    assert len(Layer(3, 4).parameters()) == 4 * (3 + 1)


def test_an_mlp_chains_its_layers() -> None:
    """3 -> 4 -> 4 -> 1, the network from the notebook."""
    model = MLP(3, [4, 4, 1])
    assert len(model.parameters()) == 4 * 4 + 4 * 5 + 1 * 5


def test_a_forward_pass_returns_one_value_per_output() -> None:
    assert len(MLP(3, [4, 2])([1.0, -2.0, 3.0])) == 2


def test_training_reduces_the_loss() -> None:
    """The whole point of the engine: gradients that point downhill."""
    random.seed(0)
    model = MLP(3, [4, 4, 1])
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    def loss() -> Value:
        predictions = [model(x) for x in xs]
        total = sum(
            (prediction - target) ** 2 for target, prediction in zip(ys, predictions, strict=True)
        )
        return total

    first = loss()
    for _ in range(20):
        total = loss()
        for parameter in model.parameters():
            parameter.grad = 0.0
        total.backward()
        for parameter in model.parameters():
            parameter.data -= 0.05 * parameter.grad

    assert loss().data < first.data * 0.5


def test_a_single_output_is_returned_unwrapped() -> None:
    """A one-neuron layer returns the value itself, which is what the training loop expects."""
    assert isinstance(MLP(3, [4, 1])([1.0, -2.0, 3.0]), Value)
