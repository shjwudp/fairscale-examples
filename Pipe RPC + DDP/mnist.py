import os
import sys
import argparse
from typing import Union, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.distributed import rpc
from torchvision import datasets, transforms
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR

import fairscale.nn.model_parallel as mpu
from fairscale.nn.pipe import PipeRPCWrapper
from fairscale.nn.pipe.types import LazyModule


_GLOBAL_ARGS = None


def get_args():
    global _GLOBAL_ARGS
    return _GLOBAL_ARGS


def get_worker_map():
    return {rank: f"Test{rank}" for rank in range(dist.get_world_size())}


class DataParallelWork:

    def __init__(self, module, dp_group) -> None:
        self.module = module
        self.dp_group = dp_group
        self._reference_global_rank = dist.distributed_c10d._get_global_rank(self.dp_group, 0)

        for t in self.module.state_dict().values():
            dist.broadcast(
                t,
                src=self._reference_global_rank,
                group=self.dp_group)

    def allreduce_gradients(self):
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.div_(self.dp_group.size())
                dist.all_reduce(p.grad, group=self.dp_group)


def setup_optimizer_and_ddp(ctx, model):
    args = get_args()

    if args.zero_stage == 0:
        model.optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.zero_stage == 1:
        base_optimizer = optim.Adadelta
        base_optimizer_arguments = dict(lr=args.lr)

        # Wrap the optimizer in its state sharding brethren
        model.optimizer = ZeroRedundancyOptimizer(
            process_group=mpu.get_data_parallel_group(),
            params=model.parameters(),
            optimizer_class=base_optimizer,
            **base_optimizer_arguments
        )
    else:
        raise NotImplementedError(f"ZeRO stage `{args.zero_stage}` not support.")

    model.scheduler = StepLR(model.optimizer, step_size=1, gamma=args.gamma)
    model.ddp = DataParallelWork(model, mpu.get_data_parallel_group())


def initialize_model_parallel_utility_interface():
    args = get_args()

    # init model parallel / data parallel process group
    if args.no_cuda:
        dist.init_process_group(backend="gloo")
    else:
        dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(
        args.model_parallel_size,
        args.pipeline_length,
        ddp_backend="gloo")
    if not args.no_cuda:
        torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())

    # init rpc
    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    rpc.init_rpc(
        f"Test{torch.distributed.get_rank()}",
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method),
    )
    if mpu.get_pipeline_parallel_group().rank() != 0:
        torch.distributed.barrier()
        rpc.shutdown()
        sys.exit(0)


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        return x


class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_model():
    sequential = [
        Net1(),
        Net2(),
        Net3(),
    ]

    return sequential


def train_test_datasets_provider():
    args = get_args()

    # Download MNIST
    if mpu.get_data_parallel_world_size() > 1:
        if mpu.get_data_parallel_rank() == 0:
            datasets.MNIST('../data', download=True)
            dist.barrier(group=mpu.get_data_parallel_group())
        else:
            dist.barrier(group=mpu.get_data_parallel_group())
    else:
        datasets.MNIST('../data', download=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if args.no_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(
        dataset1,
        sampler=DistributedSampler(
            dataset1,
            num_replicas=mpu.get_data_parallel_world_size(),
            rank=mpu.get_data_parallel_rank(),
        ),
        **train_kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset2,
        sampler=DistributedSampler(
            dataset2,
            num_replicas=mpu.get_data_parallel_world_size(),
            rank=mpu.get_data_parallel_rank(),
        ),
        **test_kwargs)

    return train_loader, test_loader


def get_pipeline_balance(model: Union[nn.Sequential, List[LazyModule]], method="uniform"):
    """Return feasible pipeline balance setting or raise exception."""
    args = get_args()
    balance = []

    if method == "uniform":
        balance = []
        n = len(model)
        while n > 0:
            balance.append(min(n, len(model) // args.pipeline_length))
            n -= len(model) // args.pipeline_length

        return balance

    raise NotImplementedError(f"Balance method {method} not implemented.")


def print_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def zero_grad(ctx, model):
    model.optimizer.zero_grad()


def step_optimizer(ctx, model):
    model.ddp.allreduce_gradients()
    model.optimizer.step()


def train(pipe, device, train_loader, epoch):
    args = get_args()

    pipe.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        pipe.foreach_worker(zero_grad, include_self=True)
        output = pipe(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        pipe.foreach_worker(step_optimizer, include_self=True)

        if batch_idx % args.log_interval == 0:
            print_rank0('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(pipe, device, test_loader):
    pipe.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = pipe(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = torch.Tensor([test_loss]).to(device)
    dist.all_reduce(test_loss, group=mpu.get_data_parallel_group())
    test_loss = test_loss.item() / len(test_loader.dataset)

    correct = torch.Tensor([correct]).to(device)
    dist.all_reduce(correct, group=mpu.get_data_parallel_group())
    correct = int(correct.item())

    print_rank0('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def step_scheduler(ctx, model):
    model.scheduler.step()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="FairScale Pipe RPC + DDP Training Example")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("--model-parallel-size", default=1, type=int)
    parser.add_argument("--pipeline-length", default=1, type=int)
    parser.add_argument("--zero-stage", default=0, choices=[0, 1])
    args = parser.parse_args()
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    args.no_cuda = args.no_cuda or not torch.cuda.is_available()

    # Initialize model parallel utility interface
    initialize_model_parallel_utility_interface()

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.cuda.current_device()

    # Initialize train/test dataloader
    train_loader, test_loader = train_test_datasets_provider()

    # Initialize PipeModule
    sequential_module = get_model()
    balance = get_pipeline_balance(sequential_module)
    pipe = PipeRPCWrapper(
        sequential_module,
        balance,
        input_device=device,
        worker_map=get_worker_map()
    )

    pipe.foreach_worker(setup_optimizer_and_ddp, include_self=True)

    # Train & Test
    for epoch in range(1, args.epochs + 1):
        train(pipe, device, train_loader, epoch)
        test(pipe, device, test_loader)
        pipe.foreach_worker(step_scheduler, include_self=True)

    torch.distributed.barrier()
    rpc.shutdown()

if __name__ == "__main__":
    main()
