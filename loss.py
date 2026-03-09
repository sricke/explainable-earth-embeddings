from satclip.satclip.loss import SatCLIPLoss
import torch.nn.functional as F
import torch.nn as nn
import torch

class ConceptLoss(SatCLIPLoss):
    def __init__(
            self,
            local_loss=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            logit_scale=1.0,
            lambda_alignment=10.0,
            sigma=1.0,
    ):
        super().__init__(local_loss, cache_labels, rank, world_size)
        self.logit_scale = logit_scale
        self.lambda_alignment = lambda_alignment
        self.sigma = sigma
        
    def contrastive_loss(self, logits_per_text, logits_per_location):
        device = logits_per_text.device
        logit_scale = self.logit_scale.exp()
        cosine_sim_text = logit_scale * logits_per_text @ logits_per_location.t()
        cosine_sim_location = cosine_sim_text.t()
        labels = self.get_ground_truth(device, cosine_sim_text.shape[0])
        return (F.cross_entropy(cosine_sim_text, labels) + F.cross_entropy(cosine_sim_location, labels)) / 2
    
    def gaussian_kernel(self, x, y):
        return torch.exp(-torch.norm(x - y, dim=1) ** 2 / (2 * self.sigma ** 2))
        
    def alignment_loss(self, logits_per_text, logits_per_location):
        return torch.mean(torch.log(self.gaussian_kernel(logits_per_text, logits_per_text))+torch.log(self.gaussian_kernel(logits_per_location, logits_per_location))-2*torch.log(self.gaussian_kernel(logits_per_text, logits_per_location)))
        
    def forward(self, logits_per_text, logits_per_location):
        contrastive_loss = self.contrastive_loss(logits_per_text, logits_per_location)
        
        alignment_loss = self.alignment_loss(logits_per_text, logits_per_location)
        return contrastive_loss + self.lambda_alignment * alignment_loss