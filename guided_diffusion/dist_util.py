"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    device_id = get_local_rank()
    torch.cuda.set_device(device_id)
    print(f"=> set cuda device = {device_id}")
    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

def get_global_rank():
    return dist.get_rank()

def get_local_rank():
    return int(os.environ["LOCAL_RANK"])

def get_world_size():
    return dist.get_world_size()

def get_local_world_size():
    return dist.get_local_world_size()

def set_local_rank(local_rank):
    os.environ["LOCAL_RANK"] = str(local_rank)

def set_fixed_environ_vars():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_PORT"] = str(_find_free_port())
    print(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])