import torch
import torch.nn as nn
import gin

@gin.register
class LogL1Loss(nn.Module):
    def __init__(self):
        super(LogL1Loss,self).__init__()
    def forward(self,activations):
        return torch.clamp(activations, min=1e-8).log().abs().mean()

@gin.register
class LogL2Loss(nn.Module):
    def __init__(self):
        super(LogL2Loss,self).__init__()
    def forward(self,activations):
        return (torch.clamp(activations, min=1e-8).log()**2).mean()

@gin.register
@gin.configurable
class PoissonLoss(nn.Module):
    def __init__(self,log_input=False,**kwargs):
        super(PoissonLoss,self).__init__()
        self.poisson = nn.PoissonNLLLoss(log_input=log_input,**kwargs)
    def forward(self,predictions,targets):
        return self.poisson(predictions,targets)

@gin.register
@gin.configurable
class BCELoss(nn.Module):
    def __init__(self,pos_weight=100,**kwargs):
        super(BCELoss,self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def forward(self,predictions,targets):
        return self.bce(predictions,targets)