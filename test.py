import time
import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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


def get_optimizer():
    supported_optimizers = {
        "adopt": ADOPT,
        "adam": optim.Adam,
        "adagrad": optim.Adagrad,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", type=str)
    parser.add_argument("lr", type=str)
    args = parser.parse_args()

    o = args.optimizer
    lr = float(args.lr)

    try:
        optimizer = supported_optimizers[o](lr)
        print(f"Testing {o} with a learning rate of {lr}")
        return optimizer
    except KeyError:
        print(
            f"Optimizer {o} is not supported, please pick one of the following:",
            ", ".join(supported_optimizers.keys()),
        )
        exit()


if __name__ == "__main__":
    num_steps = 30000

    optimizer = get_optimizer()
    X = mx.array([-1.2, 1.0])
    model = Rosenbrock(X)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    valid_loss = False
    valid_point = False
    start = time.perf_counter()
    for step in range(num_steps):
        loss, grads = loss_and_grad_fn(model)
        optimizer.update(model, grads)
        mx.eval(model, optimizer.state)
        if step % 1000 == 0:
            if (
                mx.allclose(loss, TRUE_OPTIMAL_LOSS, atol=LOSS_MARGIN)
                and not valid_loss
            ):
                print(f"Step {step}: Optimal loss converged.")
                valid_loss = True
            if (
                mx.allclose(model.X.weight, TRUE_OPTIMAL_POINT, atol=POINT_MARGIN)
                and not valid_point
            ):
                print(f"Step {step}: Optimal point converged.")
                valid_point = True
            if valid_point and valid_loss:
                taken = round(time.perf_counter() - start, 2)
                print(f"Took {taken}s")
                break

    if not valid_point or not valid_loss:
        print(f"Optimizer did not converge within {num_steps} steps.")
