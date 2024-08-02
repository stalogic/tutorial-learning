import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

NUM_EPOCHS = 100000


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
    
def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(f"cpu:{rank}")
    # ddp_model = DDP(model, device_ids=[rank])
    ddp_model = DDP(model, device_ids=None)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for _ in range(NUM_EPOCHS//world_size + 1):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(f"cpu:{rank}")
        loss_fn(outputs, labels).backward()
        optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=None)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    dist.barrier()

    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))
    ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = nn.Linear(10, 10).to(dev0)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = self.relu(self.net1(x.to(self.dev0)))
        x = self.net2(x.to(self.dev1))
        return x
    
def demo_model_parallel(rank, world_size):
    print(f"Running model parallel DDP example on rank {rank}.")
    setup(rank, world_size)

    dev0 = rank * 2
    dev1 = rank * 2 + 1
    model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def demo_default():

    model = ToyModel().to("cpu")

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for i in range(NUM_EPOCHS):
        optimizer.zero_grad()
        outputs = model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to("cpu")
        loss_fn(outputs, labels).backward()
        optimizer.step()




if __name__ == "__main__":
    world_size = 8
    run_demo(demo_basic, world_size)
    # run_demo(demo_checkpoint, world_size)
    # world_size = n_gpus // 2
    # run_demo(demo_model_parallel, world_size)

    demo_default()