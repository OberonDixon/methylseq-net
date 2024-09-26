import torch
import torch.nn as nn
import torch.nn.functional as F
import gin

@gin.configurable
@gin.register
class EncodingAdjuster(nn.Module):
    def __init__(self, encoding_str):
        super(EncodingAdjuster,self).__init__()
        self.encoding_str = encoding_str
        if self.encoding_str in ['seq+methyl_binary-seq','seq+methyl_ACGTm-sum-to-1','seq+methyl_binarize-methyl']:
            self.channels = 7
        elif self.encoding_str in ['seq+methyl_no-mask']:
            self.channels = 6
        elif self.encoding_str in ['seq+methyl_combine-strands-no-mask']:
            self.channels = 5
        elif self.encoding_str in ['seq-only']:
            self.channels = 4
        elif self.encoding_str in ['methyl-only']:
            self.channels = 1
        else:
            raise NotImplementedError(f"encoding_str: {self.encoding_str}")

    def forward(self, x):
        # the structure of the one-hot sequence is [sample,(A,C,G,T,meth_fraction_fwd,meth_fraction_rev,valid_cpg),position]

        if self.encoding_str == 'seq+methyl_binary-seq':
            x = x
        elif self.encoding_str == 'seq+methyl_no-mask':
            x =  x[:,0:6,:]
        elif self.encoding_str == 'seq+methyl_ACGTm-sum-to-1':
            x[:,1:3,:] = x[:,1:3,:] - x[:,4:6,:]
        elif self.encoding_str == 'seq+methyl_binarize-methyl':
            x[:,4:6,:] = (x[:,4:6,:]>0.5)
        elif self.encoding_str == 'seq+methyl_combine-strands-no-mask':
            x[:,4,:] = x[:,4,:] + x[:,5,:]
            x = x[:,0:5,:]
        elif self.encoding_str == 'seq-only':
            x = x[:,0:4,:]
        elif self.encoding_str == 'methyl-only':
            x = x[:,4:6,:]
        else:
            raise NotImplementedError(f"encoding_str: {self.encoding_str}")

        return x

@gin.configurable
@gin.register
class MethylationDropout(nn.Module):
    def __init__(self, in_channels, dropout=0.2):
        super(MethylationDropout, self).__init__()
        self.in_channels=in_channels
        self.dropout=dropout
    def forward(self,x):
        if self.in_channels>4 and self.training:
            num_samples, num_channels, length = x.shape
            mask = torch.rand(num_samples, length, device=x.device) > self.dropout 
            mask = mask.unsqueeze(1).expand(num_samples, self.in_channels-4, length)
            x[:,4:,:]*=mask # Zero out the methylation channels using the mask
        return x

@gin.configurable
@gin.register
class ConvDNA(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, pool_size, weight_decay=0):
        super(ConvDNA, self).__init__()
        self.in_channels=in_channels
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.conv = nn.Conv1d(in_channels, filters, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv(x)
        x = F.gelu(x)
        x = self.pool(x)
        return x

@gin.configurable
@gin.register
class ConvTower(nn.Module):
    def __init__(self, in_channels, filters_init, filters_end, divisible_by, kernel_size, pool_size, repeat, weight_decay=0):
        super(ConvTower, self).__init__()
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.repeat = repeat
        self.weight_decay = weight_decay
        self.layers = nn.ModuleList()
        filters_step = (filters_end - filters_init) // (repeat - 1) if repeat>1 else 0
        for i in range(repeat):
            filters = filters_init + i * filters_step
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size),#, padding=(kernel_size - 1) // 2),
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
@gin.register
class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dilation=1, weight_decay=0):
        super(ConvBlock, self).__init__()
        self.kernel_size=kernel_size
        self.dilation=dilation
        self.conv = nn.Conv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv(x)
        x = F.gelu(x)
        return x

@gin.configurable
@gin.register
class ConvDropout(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dropout, weight_decay=0):
        super(ConvDropout, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, filters, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x

@gin.configurable
@gin.register
class ConvFinal(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=1, shared_head=False, stride=1, weight_decay=0):
        super(ConvFinal, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = stride
        self.shared_head = shared_head # this sets the output head for all the output tracks to be the same
        self.weight_decay = weight_decay
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