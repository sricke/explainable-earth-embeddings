import torch

class SVD():
    def __init__(self, device="cuda", verbose=False, eps=1e-3, use_truncation=True):
        self.device = device
        self.verbose = verbose
        self.eps = eps
        self.use_truncation = use_truncation

    def set_C(self, C):
        self.C = C.to(self.device)

        U, S, Vh = torch.linalg.svd(self.C, full_matrices=False)

        self.U = U
        self.S = S
        self.Vh = Vh

        # ---- conditioning diagnostics ----
        cond_number = (S.max() / (S.min() + 1e-12)).item()

        print("\n[SVD DIAGNOSTIC]")
        print(f"rank = {(S > 1e-6).sum().item()} / {len(S)}")
        print(f"max singular value = {S.max().item():.4e}")
        print(f"min singular value = {S.min().item():.4e}")
        print(f"condition number   = {cond_number:.4e}")

        energy = (S**2) / (S**2).sum()
        print(f"top-5 energy share = {energy[:5].sum().item():.4f}")
        print(f"tail energy share  = {energy[-5:].sum().item():.4f}")

        # ---- STABILITY FIX ----
        self.S_safe = S.clone()

        if self.use_truncation:
            # hard cutoff (recommended for 1e10 condition numbers)
            mask = S > self.eps
            self.U = self.U[:, mask]
            self.S_safe = S[mask]
            self.Vh = self.Vh[mask, :]
        else:
            # soft regularization (ridge-style)
            self.S_safe = torch.clamp(S, min=self.eps)

        print("\n[CONDITION NUMBERS]")
        cond_orig = (S.max() / (S.min() + 1e-12)).item()
        print(f"original C:   {cond_orig:.4e}")

        if hasattr(self, "S_safe"):
            cond_eff = (self.S_safe.max() / (self.S_safe.min() + 1e-12)).item()
            print(f"effective C:  {cond_eff:.4e}")

    def fit(self, v):
        v = v.to(self.device)

        # v @ C⁺ = v @ Vh.T @ diag(1/S) @ U.T
        proj = v @ self.Vh.T

        # stable inverse (NO raw division by tiny singular values)
        proj = proj / self.S_safe

        weights = proj @ self.U.T

        return weights