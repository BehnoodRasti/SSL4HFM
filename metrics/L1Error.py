import torch
import torch.nn as nn


class L1Error(nn.Module):
    def __init__(self):
        super(L1Error, self).__init__()
        self.L1Error = nn.L1Loss()

    def forward(self, a, b):
        return self.L1Error(a, b)