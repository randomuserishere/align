import random

import numpy as np
import torch
import pandas as pd


def set_random_seed(seed: int):
    pd.options.plotting.backend = "matplotlib"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True