import time
import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ADOPT import ADOPT, ADOPTw

PI = 3.1415926535


class Test:
    def __init__(self):
        raise NotImplementedError
        # self.true_optimal_loss =
        # self.true_optimal_point =
        # self.loss_margin =
        # self.point_margin =

    def loss_fn():
        raise NotImplementedError

    def test(self, optimizer, num_steps):
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        valid_loss = False
        valid_point = False
        start = time.perf_counter()
        for step in range(num_steps):
            loss, grads = loss_and_grad_fn()
            optimizer.update(self.model, grads)
            mx.eval(self.model, optimizer.state)
            if step % 1000 == 0:
                if (
                    mx.allclose(loss, self.true_optimal_loss, atol=self.loss_margin)
                    and not valid_loss
                ):
                    print(f"Step {step}: Optimal loss converged.")
                    valid_loss = True
                if (
                    mx.allclose(
                        self.model.X.weight,
                        self.true_optimal_point,
                        atol=self.point_margin,
                    )
                    and not valid_point
                ):
                    print(f"Step {step}: Optimal point converged.")
                    valid_point = True
                if valid_point and valid_loss:
                    break

        if not valid_point or not valid_loss:
            x, y = self.model.X.weight
            ex, ey = self.true_optimal_point
            print(f"Optimizer did not converge within {num_steps} steps.")
            if not valid_point:
                print(f"Final point: ({x}, {y}). Expected ({ex}, {ey})")
            if not valid_loss:
                print(f"Final loss: {loss}. Expected {self.true_optimal_loss}")

        taken = round(time.perf_counter() - start, 2)
        print(f"Took {taken}s")


# https://www.sfu.ca/~ssurjano/rosen.html
class Rosenbrock(nn.Module):
    def __init__(self):
        super().__init__()
        self.X = nn.Linear(1, 2, False)
        self.X.weight = mx.array([-1.2, 1.0])

    def __call__(self):
        x1, x2 = self.X.weight
        return (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2


class RosenbrockTest(Test):
    def __init__(self):
        self.true_optimal_loss = mx.array(0).astype(mx.float32)
        self.true_optimal_point = mx.array([1, 1])
        self.loss_margin = 0.001
        self.point_margin = 0.01
        self.model = Rosenbrock()

    def loss_fn(self):
        return self.model()


def get_optimizer():
    supported_optimizers = {
        "adopt": ADOPT,
        "adoptw": ADOPTw,
        "adam": optim.Adam,
        "adagrad": optim.Adagrad,
        "sgd": optim.SGD,
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
    mx.random.seed(42)
    num_steps = 30000

    optimizer = get_optimizer()

    # model = Rastrigin(2)
    rosenbrokTests = RosenbrockTest()
    rosenbrokTests.test(optimizer, 23000)
