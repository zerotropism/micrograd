# micrograd

Reimplementation of @karpathy's micrograd course.

## Requirements

* Python 3.10
* run `pip install -r requirements.txt`

## Definition

* Micrograd is a light Automatic Gradient implementing backpropagation
* Backpropagation is an algorithm allowing efficient evaluation of loss function wrt the weights of a nn

Basically allows back and forth propagations through the mathematical operations chaining micrograd gets a representation of.
It implements a value class as well as methods to do so, mostly applying the chain rule from calculus then allowing to query the resulting values at steps.

Fundamentally is the only necessary piece of code to train a model, everything else is about efficiency.
