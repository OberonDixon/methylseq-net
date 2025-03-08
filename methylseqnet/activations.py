import torch
import torch.nn as nn
import gin

@gin.register
class DecadeActivation(nn.Module):
    def __init__(self):
        super(DecadeActivation, self).__init__()

    def forward(self, x):
        log_term = torch.log2(1 + torch.abs(x))
        scaled_log = torch.sign(x) * log_term
        return torch.pow(10, scaled_log)

@gin.register
class ExpActivation(nn.Module):
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)