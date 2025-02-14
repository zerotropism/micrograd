import torch

from headers import Value, Neuron, Layer, MLP
from graph import draw_dot


# Python-based construction of a neuron
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

# PyTorched-based construction of a neuron
x1 = torch.Tensor([2.0]).double()
x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()
x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()
w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()
w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double()
b.requires_grad = True
n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print("---")
print("x2", x2.grad.item())
print("w2", w2.grad.item())
print("x1", x1.grad.item())
print("w1", w1.grad.item())

# Neuron
x = [2.0, 3.0]
n = Neuron(2)
print("Neuron output", n(x))

# Layer
x = [2.0, 3.0]
n = Layer(2, 3)
print("Layer output", n(x))

# MLP
# 3-dimensional input
x = [2.0, 3.0, -1.0]
# 3 inputs into 2 layers of 4 neurons each and 1 output neuron
n = MLP(3, [4, 4, 1])
print("MLP output", n(x))
draw_dot(n(x)).render("graph", format="svg", cleanup=True, view=True)

# dataset & loss
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

# show outputs of the nn on these 4 example inputs
ypred = [n(x) for x in xs]
print("MLP starting prediction", ypred)


for k in range(20):

    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # backward pass
    # reseting all gradients first
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update weights
    for p in n.parameters():
        p.data += -0.1 * p.grad

    # print current state
    print(k, "loss =", loss.data, "predictions =", ypred)
