import torch.optim as optim
import gin

@gin.register
@gin.configurable
class AdamOptimizer(optim.Adam):
    def __init__(self, params, lr=0.005, betas=(0.97, 0.98), weight_decay=0.0, **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)

@gin.register
@gin.configurable
class SGDOptimizer(optim.SGD):
    def __init__(self, params, lr=0.005, momentum=0.98, **kwargs):
        super().__init__(params, lr=lr, momentum=momentum, **kwargs)