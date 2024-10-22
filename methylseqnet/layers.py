import torch
import torch.nn as nn
import torch.nn.functional as F
import gin

@gin.configurable
@gin.register
class EncodingAdjuster(nn.Module):
    """
    EncodingAdjuster / EncodingSelector will take a 7-dimensional input encoding ACGT-mCfrac-mGfrac-CpGmask and select a 
    different encoding for test purposes, such as seq-only, no CpGmask, or C + mCfrac add to 1.
    """
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
    def __init__(self, in_channels, dropout=0.2, chunk_size=1, inverted=False):
        super(MethylationDropout, self).__init__()
        self.in_channels=in_channels
        self.dropout=dropout
        self.chunk_size=chunk_size
        self.inverted=inverted
    def forward(self,x):
        if self.in_channels>4 and self.training:
            num_samples, num_channels, length = x.shape
            effective_length = (length + self.chunk_size - 1) // self.chunk_size
            chunk_mask = torch.rand(num_samples, effective_length, device=x.device) > self.dropout
            if self.inverted:
                chunk_mask = ~chunk_mask
            mask = chunk_mask.repeat_interleave(self.chunk_size, dim=1)
            mask = mask[:,:length]
            mask = mask.unsqueeze(1).expand(num_samples, self.in_channels-4, length)
            x[:,4:,:]*=mask # Zero out the methylation channels using the mask
        return x

@gin.configurable
@gin.register
class ConvDNA(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, pool_size, weight_decay=0, pad=False):
        super(ConvDNA, self).__init__()
        self.in_channels=in_channels
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.conv = nn.Conv1d(in_channels, filters, kernel_size, padding = (kernel_size-1)//2 if pad else 0)
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
    def __init__(self, in_channels, filters_init, filters_end, divisible_by, kernel_size, pool_size, repeat, weight_decay=0, pad=False):
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
                nn.Conv1d(in_channels, filters, kernel_size, padding=(kernel_size - 1) // 2 if pad else 0),
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
    def __init__(self, in_channels, filters, kernel_size, dilation=1, weight_decay=0, pad=False):
        super(ConvBlock, self).__init__()
        self.kernel_size=kernel_size
        self.dilation=dilation
        self.conv = nn.Conv1d(in_channels, filters, kernel_size, dilation=dilation, padding=(kernel_size -1) // 2 if pad else 0)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv(x)
        x = F.gelu(x)
        return x

@gin.configurable
@gin.register
class DilatedResidual(nn.Module):
    def __init__(
            self, 
            in_channels, 
            filters=384, 
            kernel_size=5, 
            rate_mult=1.5, 
            repeat=11, 
            dropout=0.3, 
            pad=False,
            ):
        super(DilatedResidual, self).__init__()
        self.layers = nn.ModuleList()
        self.filters = filters
        self.kernel_size = kernel_size
        self.rate_mult = rate_mult
        self.repeat = repeat
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.pad = pad

        dilation_rate = 1.0
        for _ in range(repeat):
            # First conv block with increasing dilation
            self.layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    filters, 
                    kernel_size, 
                    dilation=int(round(dilation_rate)),
                    padding=int(((kernel_size-1)// 2)*dilation_rate) 
                    if pad else 0
                ),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
            ))

            # Second conv block with residual connection (output filters match input channels)
            self.layers.append(nn.Sequential(
                nn.Conv1d(filters, in_channels, kernel_size=1),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
            ))

            # Update dilation rate
            dilation_rate *= rate_mult
            dilation_rate = round(dilation_rate)

    def forward(self, x):
        for i in range(0, len(self.layers), 2):
            residual = x
            x = self.layers[i](x)  # Dilated conv block
            x = self.layers[i + 1](x)  # 1x1 conv with residual connection

            if self.dropout:
                x = self.dropout(x)

            if residual.size(2) > x.size(2):
                crop = (residual.size(2) - x.size(2)) // 2
                residual = residual[:, :, crop:crop + x.size(2)]
            # Add residual connection
            x += residual

        return x

@gin.configurable
@gin.register
class ConvDropout(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dropout, weight_decay=0, pad=False):
        super(ConvDropout, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, filters, kernel_size, padding = (kernel_size -1)//2 if pad else 0)
        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x

@gin.configurable
@gin.register
class ConvFinal(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=1, shared_head=False, stride=1, weight_decay=0, pad=False):
        super(ConvFinal, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = stride
        self.shared_head = shared_head # this sets the output head for all the output tracks to be the same
        self.weight_decay = weight_decay
        if self.shared_head:
            self.conv = nn.Conv1d(in_channels, 1, kernel_size, stride=stride, padding=(kernel_size-1)//2 if pad else 0) # only one filter
        else:
            self.conv = nn.Conv1d(in_channels, filters, kernel_size, stride=stride, padding=(kernel_size-1)//2 if pad else 0) # multiple different output head filters

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