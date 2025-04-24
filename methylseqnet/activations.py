import torch
import torch.nn as nn
import gin

@gin.register
@gin.configurable
class ClampedReLU(nn.Module):
    def __init__(self, min_val=1e-4, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        if self.max_val is not None:
            return torch.clamp(x, min=self.min_val, max=self.max_val)
        return torch.clamp(x, min=self.min_val)

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