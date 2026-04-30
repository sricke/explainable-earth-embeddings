import torch
import torch.nn as nn


class EmbeddingQueue(nn.Module):
    """FIFO queue of raw input tensors used to extend negatives in asymmetric contrastive loss."""

    def __init__(self, queue_size: int, init: torch.Tensor):
        super().__init__()
        self.queue_size = queue_size
        self.register_buffer("queue", init)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, x: torch.Tensor):
        b = x.shape[0]
        ptr = int(self.queue_ptr)
        end = ptr + b
        if end <= self.queue_size:
            self.queue[:, ptr:end] = x.t()
        else:
            first = self.queue_size - ptr
            self.queue[:, ptr:] = x.t()[:, :first]
            self.queue[:, :b - first] = x.t()[:, first:]
        self.queue_ptr[0] = (ptr + b) % self.queue_size

    def get(self) -> torch.Tensor:  # returns (queue_size, dim)
        return self.queue.t()

class LocQueue(EmbeddingQueue):
    def __init__(self, queue_size: int, dim: int):
        if dim == 2:
            lat = torch.rand(1, queue_size) * 180 - 90   # [-90, 90]
            lon = torch.rand(1, queue_size) * 360 - 180  # [-180, 180]
            init = torch.cat([lat, lon], dim=0)
        else:
            init = torch.randn(dim, queue_size)
        super().__init__(queue_size, init)

class TextQueue(EmbeddingQueue):
    def __init__(self, queue_size: int, dim: int):
        super().__init__(queue_size, torch.randn(dim, queue_size))