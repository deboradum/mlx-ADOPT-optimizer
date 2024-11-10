# Unofficial MLX implementation of ADOPT optimizer
# https://arxiv.org/abs/2411.02853
#
#  Written by Pepijn van Wijk
# https://github.com/deboradum
#
# Apache-2.0 license

from typing import Callable, List, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


class ADOPT(Optimizer):
    r"""The ADOPT optimizer. [1].

    Implementation of ADOPT optimizer following algorithm 2 of the original
    paper.

    [1]: Shohei Taniguchi et al., 2024. ADOPT: Modified Adam Can Converge with
    Any :math:`\beta_2` with the Optimal Rate. NeurIPS 2024

    Args:
        learning_rate (float or callable): The learning rate :math:`\lambda`.
          Default: ``1e-3``
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.9999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-6``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 1e-3,
        betas: List[float] = [0.9, 0.9999],
        eps: float = 1e-6,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)
        state["t"] = 1

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the ADOPT parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        eps = self.eps

        m = state["m"]
        v = state["v"]
        t = state["t"]

        if t == 1:
            m = gradient / mx.maximum(mx.sqrt(v), eps)
        else:
            m = b1 * m + (1 - b1) * (gradient / mx.maximum(mx.sqrt(v), eps))

        v = b2 * v + (1 - b2) * mx.square(gradient)

        state["m"] = m
        state["v"] = v
        state["t"] += 1

        return parameter - lr * m


class ADOPTw(ADOPT):
    r"""The ADOPT optimizer with weight decay. [1].

    Implementation of ADOPT optimizer following algorithm 2 of the original
    paper.

    [1]: Shohei Taniguchi et al., 2024. ADOPT: Modified Adam Can Converge with
    Any :math:`\beta_2` with the Optimal Rate. NeurIPS 2024

    Args:
        learning_rate (float or callable): The learning rate :math:`\lambda`.
          Default: ``1e-3``
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.9999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-6``
        weight_decay (float, optional): Default: ``0.0``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 1e-3,
        betas: List[float] = [0.9, 0.9999],
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        super().__init__(learning_rate=learning_rate, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the ADOPT parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""

        lr = self.learning_rate.astype(gradient.dtype)
        return super().apply_single(
            gradient, parameter * (1 - lr * self.weight_decay), state
        )
