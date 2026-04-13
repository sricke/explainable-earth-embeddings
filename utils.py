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
