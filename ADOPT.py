# Unofficial MLX implementation of ADOPT optimizer
# https://arxiv.org/abs/2411.02853
#
#  Written by Pepijn van Wijk
# https://github.com/deboradum
#
# Apache-2.0 license

import math
from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
from mlx.nn import Module
from mlx.optimizers import Optimizer
from mlx.utils import tree_map, tree_reduce


class ADOPT(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 1e-3,
        betas: List[float] = [0.9, 0.9999],
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        # TODO

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the ADOPT parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        # TODO
