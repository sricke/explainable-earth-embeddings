import random
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_sched


def set_seed(seed: int):
    """
    Set global random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_scheduler(args, optimizer, total_steps: int):
    scheduler_type = getattr(args, 'scheduler', None)
    warmup_steps = getattr(args, 'warmup_steps', 0)

    if scheduler_type is None:
        return None

    T_after_warmup = total_steps - warmup_steps

    if scheduler_type == 'cosine':
        main_sched = lr_sched.CosineAnnealingLR(optimizer, T_max=T_after_warmup, eta_min=0)
    else:
        raise ValueError(f"Unknown scheduler type: '{scheduler_type}'")

    # Optionally add warmup
    if warmup_steps > 0:
        warmup = lr_sched.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
        return lr_sched.SequentialLR(
            optimizer, schedulers=[warmup, main_sched], milestones=[warmup_steps]
        )

    return main_sched


def build_optimizer(args, model, criterion):
    """
    Use AdamW optimizer with specifications on weight decay params
    """
    no_decay = lambda n, p: p.ndim < 2 or any(k in n for k in ("bn", "ln", "bias", "logit_scale"))
    named = list(model.named_parameters()) + list(criterion.named_parameters())
    return torch.optim.AdamW(
        [
            {"params": [p for n, p in named if no_decay(n, p) and p.requires_grad], "weight_decay": 0.0},
            {"params": [p for n, p in named if not no_decay(n, p) and p.requires_grad], "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
    )


class EarlyStopping:
    def __init__(self, patience_steps=5, mode="min"):
        """
        Use mode min if you want to stop on smaller values (loss), otherwise max.
        patience <= 0 disables early stopping (never returns stop=True).
        """
        self.patience_steps = patience_steps
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

        if self.patience_steps <= 0:
            return False
        return self.count >= self.patience_steps
