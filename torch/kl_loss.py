import torch
import numpy as np


prob = np.array([0.1, 0.4, 0.3, 0.2])
param = torch.tensor(np.zeros(4), dtype=torch.float32, requires_grad=True)
