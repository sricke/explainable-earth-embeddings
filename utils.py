import random
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_sched
import torch.optim.lr_scheduler as lr_sched


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_scheduler(args, optimizer):
    scheduler_type = getattr(args, 'scheduler', None)
    warmup_epochs = getattr(args, 'warmup_epochs', 0)

    if scheduler_type is None:
        return None

    T_after_warmup = args.num_epochs - warmup_epochs

    if scheduler_type == 'cosine':
        main_sched = lr_sched.CosineAnnealingLR(optimizer, T_max=T_after_warmup, eta_min=0)
    else:
        raise ValueError(f"Unknown scheduler type: '{scheduler_type}'")

    if warmup_epochs > 0:
        warmup = lr_sched.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
        )
        return lr_sched.SequentialLR(
            optimizer, schedulers=[warmup, main_sched], milestones=[warmup_epochs]
        )
    return main_sched


class EarlyStopping:
    def __init__(self, patience=5, mode="min"):
        """
        Use mode min if you want to stop on smaller values (loss), otherwise max.
        patience <= 0 disables early stopping (never returns stop=True).
        """
        self.patience = patience
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.count = 0

    def load_best(self, best_value: float) -> None:
        """Align with a resumed checkpoint so patience tracks the real best metric."""
        self.best = best_value
        self.count = 0

    def __call__(self, value: float) -> bool:
        improved = value < self.best if self.mode == "min" else value > self.best

        if improved:
            self.best = value
            self.count = 0
        else:
            self.count += 1

        if self.patience <= 0:
            return False
        return self.count >= self.patience
