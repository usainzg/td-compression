import torch
import numpy as np


def count_parameters(net: torch.nn.Module) -> np.array:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
