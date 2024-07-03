import torch
import numpy as np


prob = np.array([0.1, 0.4, 0.3, 0.2])
param = torch.tensor(np.zeros(4), dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([param], lr=0.01)
kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
while True:
    optimizer.zero_grad()
    loss = kl_loss(torch.log(torch.tensor(prob)), param)
    loss.backward()
    optimizer.step()
    print(param.data)
    if loss.data < 0.01:
        break