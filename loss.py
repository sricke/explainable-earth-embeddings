import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def make_loss(loss_type: str, logit_scale: torch.nn.Parameter, lambda_alignment: float = None, sigma: float = None) -> nn.Module:
    if loss_type == "clip":
        return CLIPLoss(logit_scale=logit_scale)
    if loss_type == "concept":
        assert lambda_alignment is not None, "lambda_alignment required for concept loss"
        assert sigma is not None, "sigma required for concept loss"
        return ConceptLoss(lambda_alignment=lambda_alignment, sigma=sigma, logit_scale=logit_scale)
    if loss_type == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss type: {loss_type}")

class CLIPLoss(nn.Module):
    """
    Forward pass uses text features and image features
    """

    def __init__(self, logit_scale: nn.Parameter = None, symmetric: bool = True):
        super().__init__()
        self.symmetric = symmetric

        if logit_scale is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = logit_scale

    def _forward_symmetric(self, text_features, location_features):
        """
        Symmetric CLIP loss like SatCLIP
        """
        device = text_features.device

        batch_size = text_features.shape[0]
        logit_scale = self.logit_scale.exp()

        # Compute scaled cosine similarity
        cosine_sim_text = text_features @ location_features.t() # (B, B) with ij entry representing how similar text i is to location j

        # Scaled logits
        logits_per_text = logit_scale * (cosine_sim_text) # (B, B) scaled
        logits_per_location = logits_per_text.T # since symmetric

        # Label matrix indicating text i should match location i and visa versa
        labels = torch.arange(batch_size, device=device)

        loss_text = F.cross_entropy(logits_per_text, labels)         # text to location
        loss_location = F.cross_entropy(logits_per_location, labels) # location to text

        total_loss = (loss_text + loss_location) / 2

        return {"contrastive_loss": total_loss}


    def _forward_asymmetric(self, text_features, location_features, gps_embeddings_queue: torch.Tensor = None):
        """
        Asymmetric CLIP loss like GeoCLIP with GPS Queue
        """
        device = text_features.device

        batch_size = text_features.shape[0]
        logit_scale = self.logit_scale.exp()

        assert gps_embeddings_queue is not None, "Must provide gps queue"
        assert gps_embeddings_queue.shape[1] == location_features.shape[1], "gps embeddings and location embeddings need to have the same dimension"

        cosine_sim_location_to_text = text_features @ location_features.t() # (B, B) with ij entry representing how similar text i is to location j

        # Scaled logits
        logits_location_to_text = logit_scale * cosine_sim_location_to_text

        gps_embeddings_queue = F.normalize(gps_embeddings_queue, dim=1) # Normalize embeddings in gps queue

        all_location_features = torch.cat([location_features, gps_embeddings_queue])
        cosine_sim_all_text = text_features @ all_location_features.t() # (B, dim) * (dim, B + Q) = (B, B+Q)
        logits_text_to_all_locations = logit_scale * (cosine_sim_all_text) 

        labels = torch.arange(batch_size, device=device)
        loss_text = F.cross_entropy(logits_text_to_all_locations, labels)         # text to location
        loss_location = F.cross_entropy(logits_location_to_text, labels) # location to text

        total_loss = (loss_text + loss_location) / 2

        return {"contrastive_loss": total_loss}

    def forward(self, text_features, location_features, output_dict=False, gps_queue: torch.Tensor = None):
        """
        Input:
            - text_features: (B, text_emb_dim)
            - location_features: (B, loc_emb_dim)
        """
        if self.symmetric:
            loss_dict = self._forward_symmetric(text_features, location_features)
            return loss_dict if output_dict else loss_dict["contrastive_loss"]
        else:
            loss_dict = self._forward_asymmetric(text_features, location_features, gps_queue = gps_queue)
            return loss_dict if output_dict else loss_dict["contrastive_loss"]
        

class ConceptLoss(nn.Module):

    def __init__(self, lambda_alignment: float = 0.0, sigma: float = 1.0, logit_scale: nn.Parameter = None, clip_symmetric: bool = True):
        super().__init__()
        self.lambda_alignment = lambda_alignment
        self.sigma = sigma

        self.clip_loss = CLIPLoss(logit_scale=logit_scale, symmetric=clip_symmetric)

    def _gaussian_kernel_matrix(self, A, B):
        """
        Input:
            - A: (N, D)
            - B: (M, D)

        Returns:
            - Kernel matrix of size (N, M)
        """
        dist2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)  # (N, M)
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