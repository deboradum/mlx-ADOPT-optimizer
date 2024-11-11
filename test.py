import time

import mlx.core as mx
import mlx.nn as nn

from ADOPT import ADOPT

TRUE_OPTIMAL_LOSS = mx.array(0).astype(mx.float32)
TRUE_OPTIMAL_POINT = mx.array([1, 1])
LOSS_MARGIN = 0.001
POINT_MARGIN = 0.01

# https://www.sfu.ca/~ssurjano/rosen.html
class Rosenbrock(nn.Module):
    def __init__(self, X):
        super().__init__()
        self.X = nn.Linear(1, 2, False)
        self.X.weight = X

    def __call__(self):
        x1, x2 = self.X.weight
        return (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2


def loss_fn(model):
    return model()

if __name__ == "__main__":
    optimizer = ADOPT(1e-3)

    X = mx.array([-1.2, 1.0])
    model = Rosenbrock(X)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    start = time.perf_counter()
    for step in range(23000):
        loss, grads = loss_and_grad_fn(model)
        optimizer.update(model, grads)
        mx.eval(model, optimizer.state)
    taken = round(time.perf_counter()-start, 2)
    print(f"Took {taken}s")

    optimal_point = model.X.weight
    optimal_loss = model()

    assert mx.allclose(
        optimal_loss, TRUE_OPTIMAL_LOSS, atol=LOSS_MARGIN
    ), f"Optimal loss value is larger than expected. Got {optimal_loss}, expected {TRUE_OPTIMAL_LOSS}"
    print("Optimal loss is accurate.")
    assert mx.allclose(
        optimal_point, TRUE_OPTIMAL_POINT, atol=POINT_MARGIN
    ), f"Optimal point is not accurate. Got {optimal_point}, expected {TRUE_OPTIMAL_POINT}"
    print("Optimal point is accurate.")

    print("Valid!")
