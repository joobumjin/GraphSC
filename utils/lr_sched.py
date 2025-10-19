####
# This code was directly repurposed from 
# https://github.com/facebookresearch/mae
####

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

class HalfCosDecay:
    def __init__(self, warmup_epochs: Optional[int] = 10, max_epochs: Optional[int] = 100,
                 min_lr: Optional[float] = 0.0, start_lr: Optional[float] = 1e-3):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.start_lr = start_lr

    def adjust_learning_rate(self, optimizer, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.start_lr * epoch / self.warmup_epochs 
        else:
            lr = self.min_lr + (self.start_lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
