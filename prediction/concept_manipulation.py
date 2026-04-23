import numpy as np

from .eval import eval_ridge


def run_manipulation(task, probe, codes_test, y_test, codes_train, top_k_idx, random_k_idx=None):
    """
    Evaluate a frozen probe under three concept manipulations of top_k_idx,
    plus an optional random-ablation control.

    Returns dict: {condition: metric_value}
    """
    p99 = np.percentile(codes_train, 99, axis=0)

    def apply(codes, idx, condition):
        z = codes.copy()
        if condition == "zeroed":
            z[:, idx] = 0.0
        elif condition == "maximized":
            z[:, idx] = p99[idx]
        return z

    results = {
        cond: eval_ridge(task, probe, apply(codes_test, top_k_idx, cond), y_test)
        for cond in ("original", "zeroed", "maximized")
    }
    if random_k_idx is not None:
        results["random_zeroed"] = eval_ridge(task, probe, apply(codes_test, random_k_idx, "zeroed"), y_test)
    return results
