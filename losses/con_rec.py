import torch
import torch.nn as nn   

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import NTXentLoss
from lightly.loss import BarlowTwinsLoss
from lightly.loss import VICRegLoss
class ConRecLoss(nn.Module):
    """
    Combines contrastive loss and reconstruction loss.
    Contrastive loss aligns representations.
    Reconstruction loss ensures faithful reconstructions.
    """
    def __init__(self):
        super().__init__()
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    def forward(self, pred, target,  z1, z2, x_encoded_masked=None,temperature=0.5, l1_lambda=0.1, l2_lambda=0.5):
        """
        z1, z2: [batch_size, embedding_dim]
        pred, target: [batch_size, num_masked_tokens, patch_dim]
        """
        contrastive = BarlowTwinsLoss()
        reconstruction = self.l2_loss(pred, target)
        # Optional latent regularization
        if x_encoded_masked is not None:
            latent_reg_loss = x_encoded_masked.abs().mean()
        else:
            latent_reg_loss = 0.0  # Safe fallback
        return reconstruction + l2_lambda * contrastive(z1,z2) #+  l1_lambda * latent_reg_loss# self.l1_loss(x_encoded_masked, torch.zeros_like(x_encoded_masked))

    # def forward(self, pred, target, x_encoded_masked, z1, z2, temperature=0.5, l1_lambda=0.1, l2_lambda=0.5):
    #     """
    #     z1, z2: [batch_size, embedding_dim]
    #     pred, target: [batch_size, num_masked_tokens, patch_dim]
    #     """
    #     contrastive = BarlowTwinsLoss(lambda_param=temperature)
    #     reconstruction = self.l2_loss(pred, target)
    #     return reconstruction + l2_lambda * contrastive(z1,z2) +  l1_lambda * self.l1_loss(x_encoded_masked, torch.zeros_like(x_encoded_masked))

# def contrastive_loss(z1, z2, temperature=0.1):
#     z1 = F.normalize(z1, dim=-1)
#     z2 = F.normalize(z2, dim=-1)
#     logits = torch.matmul(z1, z2.T) / temperature
#     labels = torch.arange(z1.size(0), device=z1.device)
#     return F.cross_entropy(logits, labels)


