import torch
import numpy as np

prob = torch.tensor(np.array([[1, 4, 5, 3]]), dtype=torch.float32)
model = torch.nn.Linear(4, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(50):
    logit = model(torch.tensor(np.array([[1, 2, 3, 4]]), dtype=torch.float32))
    pred = torch.nn.functional.softmax(logit, dim=-1)
    # loss = torch.nn.functional.kl_div(pred.log(), prob)
    loss = torch.nn.functional.cross_entropy(logit, prob)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    print(f"{loss.item()=}, {pred=}, {prob=}")