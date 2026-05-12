import math

import torch
import torch.nn.functional as F
import torch.nn as nn

class CLIPLoss(nn.Module):
    def __init__(
        self,
        logit_scale_init: float = math.log(1 / 0.07)
    ):
        super().__init__()
        # Logit scale is learnable parameter
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))

    def forward(self, text_features, location_features):
        """
        Symmetric CLIP loss w.r.t text and location features
        """
        device = text_features.device
        batch_size = text_features.shape[0]
        logit_scale = self.logit_scale.exp()

        cosine_sim_text = text_features @ location_features.t()
        logits_per_text = logit_scale * cosine_sim_text
        logits_per_location = logits_per_text.T

        labels = torch.arange(batch_size, device=device)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss_location = F.cross_entropy(logits_per_location, labels)

        return (loss_text + loss_location) / 2