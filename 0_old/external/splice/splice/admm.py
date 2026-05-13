import torch

class ADMM():
    def __init__(self, rho=1., l1_penalty=0.2, tol=1e-4, max_iter=10000, device="cuda", verbose=False, check_every=10):
        self.rho = rho
        self.l1_penalty = l1_penalty
        self.tol = tol
        self.max_iter = max_iter
        self.device = device
        self.verbose = verbose
        self.check_every = check_every

    def set_C(self, C):
        self.C = C

        c = C.shape[0]
        self.c = c

        Q = 2*C @ C.T + torch.eye(c, device=self.device)*self.rho
        self.Q = 2*C @ C.T + torch.eye(c, device=self.device)*self.rho
        self.Q_cho = torch.linalg.cholesky(Q)

    def step(self, Cb, Q_cho, z, u):
        xn = torch.cholesky_solve(2*Cb + self.rho*(z - u), Q_cho)
        zn = torch.where((xn + u - self.l1_penalty/self.rho) > 0, xn + u - self.l1_penalty/self.rho, 0)
        un = u + xn - zn
        return xn, zn, un

    def fit(self, v):
        # precompute once — C and v don't change across iterations
        Cb = self.C @ v.T

        # iterates, size: c x batch
        x = torch.randn((self.c, v.shape[0])).to(self.device)
        z = torch.randn((self.c, v.shape[0])).to(self.device)
        u = torch.randn((self.c, v.shape[0])).to(self.device)

        res_prim = res_dual = None
        for ix in range(self.max_iter):
            z_old = z
            x, z, u = self.step(Cb, self.Q_cho, z, u)

            # .max() forces a GPU→CPU sync; only pay that cost every check_every steps
            if ix % 10 == 0:
                res_prim = torch.linalg.norm(x-z, dim=0)
                res_dual = torch.linalg.norm(self.rho*(z-z_old), dim=0)

                if (res_prim.max() < self.tol) and (res_dual.max() < self.tol):
                    break
        if self.verbose:
            print("Stopping at iteration {}".format(ix))
            if res_prim is not None:
                print("Prime Residual, r_k: {}".format(res_prim.mean()))
                print("Dual Residual, s_k: {}".format(res_dual.mean()))
        return z.T