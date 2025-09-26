import torch
import torch.nn as nn

class MSEPosWeightLoss(nn.Module):
    def __init__(self, lambda_reg=1e-1):
        super(MSEPosWeightLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # Initialize MSE loss
        self.lambda_reg = lambda_reg  # Regularization coefficient

    def forward(self, output, target, model):
        # Compute the MSE loss
        loss = self.mse_loss(output, target)
        
        # Add regularization term for negative weights
        for name, param in model.named_parameters():
            if "decoder_conv.weight" in name:
                loss += self.lambda_reg * torch.sum(torch.clamp(param, max=0)**2)  # Penalize negative weights
        
        return loss