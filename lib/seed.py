import os
import random

import numpy
import torch


class RNG:
    def __init__(self):
        pass

    def is_deterministic(self):
        return self.seed is not None

    def set_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def disable_determinism(self):
        torch.seed()
        torch.cuda.seed_all()
        random.seed()
        numpy.random.seed(None)
        os.environ.pop("PYTHONHASHSEED", None)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def get_torch_generator(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return g
