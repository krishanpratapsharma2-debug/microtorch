from engine import Value
from nn import MLP

n = MLP(3, [4, 4, 1])

xs = [
    [Value(1.0), Value(2.0), Value(3.0)],
    [Value(4.0), Value(5.0), Value(6.0)],
    [Value(7.0), Value(8.0), Value(9.0)],
]

ys = [Value(1.0), Value(0.0), Value(-1.0)]

for k in range(1000):

    # forward
    ypred = [n(x) for x in xs]

    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # zero grad
    for p in n.parameters():
        p.grad = 0

    # backward
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.1 * p.grad

    if k in [100, 300, 500, 700, 999]:
        print(k, loss.data)
