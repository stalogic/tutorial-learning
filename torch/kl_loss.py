import torch
import numpy as np

prob = torch.tensor(np.array([0.1, 0.4, 0.3, 0.2]), dtype=torch.float32)
model = torch.nn.Sequential(
    torch.nn.Linear(4, 4),
    torch.nn.Softmax(dim=-1)
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print(f"{optimizer=}")
for _ in range(200):
    pred = model(torch.tensor(np.array([[1, 2, 3, 4]]), dtype=torch.float32))
    loss = torch.nn.functional.kl_div(pred.log(), prob)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    print(f"{loss.item()=}, {pred=}, {prob=}")