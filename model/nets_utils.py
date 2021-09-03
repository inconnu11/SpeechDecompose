# -*- coding: utf-8 -*-

"""Network related utility tools."""

import logging
import numpy as np
import torch

def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
    