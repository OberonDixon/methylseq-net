import torch
import torch.nn as nn
import torch.nn.functional as F
import gin

@gin.configurable
class ConvDNA(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, pool_size):
        super(ConvDNA, self).__init__()
        self.conv = nn.Conv1d(in_channels, filters, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.in_channels = in_channels

    def forward(self, x):
        # the structure of the one-hot sequence is [sample,(A,C,G,T,meth_fraction,valid_cpg),position]
        
        # If in_channels in only 4, and the input data has cpg methylation, we want to trim that off!
        # On the other hand, if in_channels is 1, we *only* keep cpg methylation, and if it is 2 we keep cpg methylation
        # and valid cpgs
        if x.shape[1] > self.in_channels:
            if self.in_channels>2:
                x = x[:, :self.in_channels, :]
            elif self.in_channels>0:
                x = x[:, -2:(-2+self.in_channels), :]
            else:
                raise ValueError(f"in_channels {self.in_channels} invalid value.")
        x = self.conv(x)
        x = F.gelu(x)
        x = self.pool(x)
        return x

@gin.configurable
class ConvTower(nn.Module):
    def __init__(self, in_channels, filters_init, filters_end, divisible_by, kernel_size, pool_size, repeat):
        super(ConvTower, self).__init__()
        self.layers = nn.ModuleList()
        filters_step = (filters_end - filters_init) // (repeat - 1) if repeat>1 else 0
        for i in range(repeat):
            filters = filters_init + i * filters_step
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(filters),
                nn.GELU(),
                nn.MaxPool1d(pool_size)
            ))
            in_channels = filters

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

@gin.configurable
class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, filters, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = F.gelu(x)
        return x

@gin.configurable
class ConvDropout(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dropout):
        super(ConvDropout, self).__init__()
        self.conv = nn.Conv1d(in_channels, filters, kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x

@gin.configurable
class ConvFinal(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=1, shared_head=False, stride=1):
        super(ConvFinal, self).__init__()
        self.filters = filters
        self.shared_head = shared_head # this sets the output head for all the output tracks to be the same
        if self.shared_head:
            self.conv = nn.Conv1d(in_channels, 1, kernel_size, stride=stride) # only one filter
        else:
            self.conv = nn.Conv1d(in_channels, filters, kernel_size, stride=stride) # multiple different output head filters

    def forward(self, x):
        x = self.conv(x)
        if self.shared_head:
            x = x.repeat(1, self.filters, 1) # duplicate output value across all tracks
        return x

# class DenseBlock(nn.Module):
#     def __init__(self, in_features, units, dropout):
#         super(DenseBlock, self).__init__()
#         self.fc = nn.Linear(in_features, units)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.fc(x)
# #         x = F.gelu(x)
#         x = self.dropout(x)
#         return x

# class Final(nn.Module):
#     def __init__(self, in_features, units):
#         super(Final, self).__init__()
#         self.fc = nn.Linear(in_features, units)

#     def forward(self, x):
#         x = self.fc(x)
#         # x = torch.sigmoid(x)
#         return x