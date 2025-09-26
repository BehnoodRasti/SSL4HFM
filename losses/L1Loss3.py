import torch
import torch.nn as nn

class L1SparsLoss(nn.Module):
    def __init__(self):
        super(L1SparsLoss, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()


    def forward(self, target, pred,x_encoded_masked,l1_lambda):
        """
        Compute the L1 loss between target and pred.

        Args:
            target (Tensor): The ground truth tensor.
            pred (Tensor): The predicted tensor.

        Returns:
            Tensor: The computed L1 loss.
        """
        return self.l2_loss(target, pred) + l1_lambda * self.l1_loss(x_encoded_masked, torch.zeros_like(x_encoded_masked))        