# MicroTorch

A minimal PyTorch-inspired deep learning framework built completely from scratch in Python.

MicroTorch implements reverse-mode automatic differentiation, computational graphs, backpropagation, and a small neural network library without using any deep learning frameworks.

---

## Features

- Reverse-mode automatic differentiation
- Dynamic computational graphs
- Backpropagation engine
- Operator overloading
- Tanh activation function
- Multi Layer Perceptron (MLP)
- Gradient descent training
- Neural network training from scratch

---

## Project Structure

```text
microtorch/
│
├── engine.py      # Autograd engine and Value class
├── nn.py          # Neuron, Layer, and MLP implementations
├── train.py       # Example training script
├── README.md
└── LICENSE
```

---

## Example

### Define a Neural Network

```python
mlp = MLP(3, [4, 4, 1])
```

### Forward Pass

```python
ypred = [mlp(x) for x in xs]
```

### Loss Computation

```python
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
```

### Backpropagation

```python
loss.backward()
```

### Gradient Descent Update

```python
for p in mlp.parameters():
    p.data += -0.01 * p.grad
```

---

## Training Example

```python
xs = [
    [Value(1.0), Value(2.0), Value(3.0)],
    [Value(4.0), Value(5.0), Value(6.0)],
    [Value(7.0), Value(8.0), Value(9.0)],
]

ys = [Value(1.0), Value(0.0), Value(-1.0)]
```

---

## Concepts Implemented

- Chain Rule
- Reverse Mode Autodiff
- Computational Graphs
- Backpropagation
- Neural Networks
- Gradient Descent
- Parameter Optimization

---

## Future Improvements

- ReLU / Sigmoid activations
- Adam optimizer
- Tensor support
- Mini-batch training
- GPU acceleration
- Dropout and normalization layers

---

## Inspiration

Inspired by:
- PyTorch
- Andrej Karpathy's Micrograd

---

## License

This project is licensed under the MIT License.
