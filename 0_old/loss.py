import math

import torch
import torch.nn.functional as F
import torch.nn as nn


def make_loss(
    loss_type: str,
    logit_scale_temp: float = 0.07,
    lambda_alignment: float = None,
    sigma: float = None,
    use_layer_norm: bool = False,
    whiten: bool = False,
    learnable_temp: bool = True,
) -> nn.Module:
    logit_scale_init = math.log(1.0 / logit_scale_temp)
    kwargs = dict(use_layer_norm=use_layer_norm, whiten=whiten, learnable_temp=learnable_temp)
    if loss_type == "clip_symmetric":
        return CLIPLoss(logit_scale_init=logit_scale_init, symmetric=True, **kwargs)
    if loss_type == "clip_asymmetric":
        return CLIPLoss(logit_scale_init=logit_scale_init, symmetric=False, **kwargs)
    if loss_type == "concept":
        assert lambda_alignment is not None, "lambda_alignment required for concept loss"
        assert sigma is not None, "sigma required for concept loss"
        return ConceptLoss(lambda_alignment=lambda_alignment, sigma=sigma, logit_scale_init=logit_scale_init, **kwargs)
    if loss_type == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss type: {loss_type}")


def _whiten(x: torch.Tensor) -> torch.Tensor:
    """ZCA whitening: decorrelates features and scales to unit variance across the batch."""
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    U, S, _ = torch.linalg.svd(cov)
    W = U @ torch.diag(1.0 / (S.sqrt() + 1e-5)) @ U.T
    return x @ W.T


class CLIPLoss(nn.Module):
    def __init__(
        self,
        logit_scale_init: float = math.log(1 / 0.07),
        symmetric: bool = True,
        use_layer_norm: bool = False,
        whiten: bool = False,
        learnable_temp: bool = True,
    ):
        super().__init__()
        self.symmetric = symmetric
        self.use_layer_norm = use_layer_norm
        self.whiten = whiten
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))
        else:
            self.register_buffer("logit_scale", torch.tensor(logit_scale_init))

    def _preprocess(self, text_features: torch.Tensor, location_features: torch.Tensor):
        if self.use_layer_norm:
            text_features = F.layer_norm(text_features, text_features.shape[-1:])
            location_features = F.layer_norm(location_features, location_features.shape[-1:])
        if self.whiten:
            text_features = _whiten(text_features)
            location_features = _whiten(location_features)
        # Re-normalize to unit sphere after any preprocessing so dot-product == cosine similarity.
        text_features = F.normalize(text_features, dim=-1)
        location_features = F.normalize(location_features, dim=-1)
        return text_features, location_features

    def _forward_symmetric(self, text_features, location_features):
        """Symmetric CLIP loss like SatCLIP"""
        device = text_features.device
        batch_size = text_features.shape[0]
        logit_scale = self.logit_scale.exp()

        cosine_sim_text = text_features @ location_features.t()
        logits_per_text = logit_scale * cosine_sim_text
        logits_per_location = logits_per_text.T

        labels = torch.arange(batch_size, device=device)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss_location = F.cross_entropy(logits_per_location, labels)

        return {"contrastive_loss": (loss_text + loss_location) / 2}

    def _forward_asymmetric(self, text_features, location_features, loc_queue: torch.Tensor = None, text_queue: torch.Tensor = None):
        """Asymmetric CLIP loss: extend negatives on the location side (loc_queue) or text side (text_queue)."""
        device = text_features.device
        batch_size = text_features.shape[0]
        logit_scale = self.logit_scale.exp()

        assert (loc_queue is None) != (text_queue is None), "Provide exactly one of loc_queue or text_queue"

        if loc_queue is not None:
            assert loc_queue.shape[1] == location_features.shape[1], "loc_queue dim must match location_features dim"
            all_location_features = torch.cat([location_features, F.normalize(loc_queue, dim=1)])
            logits = logit_scale * (text_features @ all_location_features.t())
            labels = torch.arange(batch_size, device=device)
            loss = F.cross_entropy(logits, labels)
        else:
            assert text_queue.shape[1] == text_features.shape[1], "text_queue dim must match text_features dim"
            all_text_features = torch.cat([text_features, F.normalize(text_queue, dim=1)])
            logits = logit_scale * (location_features @ all_text_features.t())
            labels = torch.arange(batch_size, device=device)
            loss = F.cross_entropy(logits, labels)

        return {"contrastive_loss": loss}

    def forward(self, text_features, location_features, output_dict=False, loc_queue: torch.Tensor = None, text_queue: torch.Tensor = None):
        """
        Input:
            - text_features: (B, text_emb_dim)
            - location_features: (B, loc_emb_dim)
        """
        text_features, location_features = self._preprocess(text_features, location_features)
        if self.symmetric:
            loss_dict = self._forward_symmetric(text_features, location_features)
        else:
            loss_dict = self._forward_asymmetric(text_features, location_features, loc_queue=loc_queue, text_queue=text_queue)
        return loss_dict if output_dict else loss_dict["contrastive_loss"]


class ConceptLoss(nn.Module):

    def __init__(
        self,
        lambda_alignment: float = 0.0,
        sigma: float = 1.0,
        logit_scale_init: float = math.log(1 / 0.07),
        clip_symmetric: bool = True,
        use_layer_norm: bool = False,
        whiten: bool = False,
        learnable_temp: bool = True,
    ):
        super().__init__()
        self.lambda_alignment = lambda_alignment
        self.sigma = sigma
        self.clip_loss = CLIPLoss(
            logit_scale_init=logit_scale_init,
            symmetric=clip_symmetric,
            use_layer_norm=use_layer_norm,
            whiten=whiten,
            learnable_temp=learnable_temp,
        )

    def _gaussian_kernel_matrix(self, A, B):
        """
        Input:
            - A: (N, D)
            - B: (M, D)
        Returns:
            - Kernel matrix of size (N, M)
        """
        dist2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return torch.exp(-dist2 / (2 * self.sigma ** 2))

    def _alignment_loss(self, text_features, location_features):
        assert text_features.shape[0] == location_features.shape[0], "Need to have the same batch size"
        batch_size = text_features.shape[0]

        K_tt = self._gaussian_kernel_matrix(text_features, text_features)
        K_ll = self._gaussian_kernel_matrix(location_features, location_features)
        K_tl = self._gaussian_kernel_matrix(text_features, location_features)

        loss = (torch.log(K_tt).sum() +
                torch.log(K_ll).sum() -
                2 * torch.log(K_tl).sum()) / (batch_size**2)
        return loss

    def forward(self, text_features, location_features):
        return self.clip_loss(text_features, location_features) + self.lambda_alignment * self._alignment_loss(text_features, location_features)
