import torch
import random
import numpy as np


def set_torch_deterministic(random_state: int = 42, benchmark=False) -> None:
    random_state = int(random_state)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark
        torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)
