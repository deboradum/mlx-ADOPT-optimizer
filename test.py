import time
import argparse

import numpy as np
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

    def loss_fn(self):
        return self.model()

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
            print(f"Optimizer did not converge within {num_steps} steps.")

            point = [str(x) for x in np.array(self.model.X.weight)]
            opt_point = [str(x) for x in np.array(self.true_optimal_point)]

            if not valid_point:
                print(
                    f"Final point: ({', '.join(str(x) for x in point)}). Expected ({', '.join(opt_point)})",
                )
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


# https://www.sfu.ca/~ssurjano/rastr.html
class Rastrigin(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        init_fn = nn.init.uniform(-5.12, 5.12)
        self.X = nn.Linear(1, n, False)
        self.X.weight = init_fn(mx.zeros_like(self.X.weight))

    def __call__(self):
        return 10 * self.n + mx.sum(
            mx.square(self.X.weight) - 10 * mx.cos(2 * PI * self.X.weight)
        )


class RastriginTest(Test):
    def __init__(self, n):
        self.true_optimal_loss = mx.array(0).astype(mx.float32)
        self.true_optimal_point = mx.zeros(n)
        self.loss_margin = 0.001
        self.point_margin = 0.01
        self.model = Rastrigin(4)


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
    rosenbrokTests = RosenbrockTest()
    rosenbrokTests.test(optimizer, 20000)

    optimizer = get_optimizer()
    rastriginTest = RastriginTest(5)
    rastriginTest.test(optimizer, 20000)