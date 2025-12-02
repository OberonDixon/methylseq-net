import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint
import gin
import warnings

from borzoi_pytorch.pytorch_borzoi_transformer import Attention, FlashAttention
from borzoi_pytorch.pytorch_borzoi_utils import Residual

gin.external_configurable(nn.AvgPool1d, module='torch.nn')
gin.external_configurable(nn.MaxPool1d, module='torch.nn')

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for adversarial training.
    
    Passes input unchanged in forward pass, but scales gradients in backward pass.
    
    Args:
        lambda_: Scaling factor for gradients:
            - Positive values: Normal gradient flow (scaled by lambda)
            - Negative values: Reversed gradients (adversarial training)
            - Zero: No gradients flow through
            - Default: 0.0 (stop gradients)
    """
    def __init__(self, lambda_=0.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)

@gin.configurable
@gin.register
class ActivationCapture(nn.Module):
    """Dummy module to enable hook registration on non-module operations."""
    def forward(self, x):
        return x

@gin.configurable
@gin.register
class EncodingAdjuster(nn.Module):
    """
    OBSOLETE: this class is obsolete. Use transforms.py::EncodingSelector instead.
    EncodingAdjuster / EncodingSelector will take a 7-dimensional input encoding ACGT-mCfrac-mGfrac-CpGmask and select a 
    different encoding for test purposes, such as seq-only, no CpGmask, or C + mCfrac add to 1.
    """
    def __init__(self, encoding_str):
        super().__init__()
        warnings.warn(
            "The EncodingAdjuster class in layers.py is deprecated and will be removed in a future release. "
            "Please use transforms.py::EncodingSelector instead.",
            UserWarning,
            stacklevel=2
        )
        self.encoding_str = encoding_str
        if self.encoding_str in ['seq+methyl_binary-seq','seq+methyl_ACGTm-sum-to-1','seq+methyl_binarize-methyl']:
            self.channels = 7
        elif self.encoding_str in ['seq+methyl_no-mask']:
            self.channels = 6
        elif self.encoding_str in ['seq+methyl_combine-strands-no-mask','seq+smoothed-methyl']:
            self.channels = 5
        elif self.encoding_str in ['seq-only']:
            self.channels = 4
        elif self.encoding_str in ['methyl-only']:
            self.channels = 3
        elif self.encoding_str in ['smoothed-methyl-only']:
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
            x = x.clone()
            x[:,1:3,:] = x[:,1:3,:] - x[:,4:6,:]
        elif self.encoding_str == 'seq+methyl_binarize-methyl':
            x = x.clone()
            x[:,4:6,:] = (x[:,4:6,:]>0.5)
        elif self.encoding_str == 'seq+methyl_combine-strands-no-mask':
            x = x.clone()
            x[:,4,:] = x[:,4,:] + x[:,5,:]
            x = x[:,0:5,:]
        elif self.encoding_str == 'seq+smoothed-methyl':
            mask = x[:,6,:]>0
            methylation = x[:,4,:]+x[:,5,:]
            padding = (129 - 1) // 2
            kernel = torch.ones(1, 1, 129, device=methylation.device)  # Create a kernel with ones
            smoothed = F.conv1d(methylation.unsqueeze(1), kernel, padding=padding).squeeze(1)  # Apply convolution
            mask_sum = F.conv1d(mask.unsqueeze(1).float(), kernel, padding=padding).squeeze(1)
            smoothed = torch.nan_to_num((smoothed / mask_sum),nan=1,posinf=1)
            x = x.clone()
            x[:,4,:] = smoothed
            x = x[:,0:5,:]
        elif self.encoding_str == 'seq-only':
            x = x[:,0:4,:]
        elif self.encoding_str == 'methyl-only':
            x = x[:,4:7,:]
        elif self.encoding_str == 'smoothed-methyl-only':
            mask = x[:,6,:]>0
            methylation = x[:,4,:]+x[:,5,:]
            padding = (129 - 1) // 2
            kernel = torch.ones(1, 1, 129, device=methylation.device)  # Create a kernel with ones
            smoothed = F.conv1d(methylation.unsqueeze(1), kernel, padding=padding).squeeze(1)  # Apply convolution
            mask_sum = F.conv1d(mask.unsqueeze(1).float(), kernel, padding=padding).squeeze(1)
            smoothed = torch.nan_to_num((smoothed / mask_sum),nan=1,posinf=1)
            x = x.clone()
            x[:,4,:] = smoothed
            x = x[:,4:5,:]
        else:
            raise NotImplementedError(f"encoding_str: {self.encoding_str}")

        return x

@gin.configurable
@gin.register
class MethylationDropout(nn.Module):
    def __init__(self, in_channels, dropout=0.2, chunk_size=1, inverted=False):
        super().__init__()
        warnings.warn(
            "The MethylationDropout class in layers.py is deprecated and will be removed in a future release. "
            "Please use transforms.py::CpGSparsifier instead.",
            UserWarning,
            stacklevel=2
        )
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
    def __init__(self, in_channels, filters, kernel_size, pool_size, weight_decay=0, pad=False, stride=1, pool_class=nn.MaxPool1d):
        super().__init__()
        self.in_channels=in_channels
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.conv = nn.Conv1d(
            in_channels, 
            filters, 
            kernel_size, 
            padding = (kernel_size-1)//2 if pad else 0, 
            stride=stride
        )
        self.pool = pool_class(pool_size)
        self.weight_decay = weight_decay
        self.stride=stride

    def forward(self, x):
        x = self.conv(x)
        x = F.gelu(x)
        x = self.pool(x)
        return x

@gin.configurable
@gin.register
class ConvTower(nn.Module):
    def __init__(self, in_channels, filters_init, filters_end, divisible_by, kernel_size, pool_size, repeat, weight_decay=0, pad=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.repeat = repeat
        self.weight_decay = weight_decay
        self.layers = nn.ModuleList()
        filters_step = (filters_end - filters_init) // (repeat - 1) if repeat>1 else 0
        for i in range(repeat):
            if i<repeat-1:
                filters = filters_init + i * filters_step
            else:
                filters = filters_end
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
    def __init__(self, in_channels, filters, kernel_size, pool_size=1, pool_class=nn.MaxPool1d, dilation=1, weight_decay=0, pad=False):
        super().__init__()
        self.kernel_size=kernel_size
        self.dilation=dilation
        self.pool_size=pool_size
        self.conv = nn.Conv1d(in_channels, filters, kernel_size, dilation=dilation, padding=(kernel_size -1) // 2 if pad else 0)
        self.weight_decay = weight_decay
        self.pool = pool_class(pool_size) if pool_size>1 else None

    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
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
        super().__init__()
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
            x = x + residual

        return x

@gin.configurable
@gin.register
class BorzoiTransformerBlock(nn.Module):
    """
    A single Borzoi-style transformer block with self-attention and FFN.
    Implementation follows the pattern here: https://github.com/johahi/borzoi-pytorch/blob/main/borzoi_pytorch/pytorch_borzoi_model.py#L106

    The following modifications have been made compared to what is described for Borzoi:
     -  we maintain a (batch, channels, seq_len) interface to be consistent with Conv layers,
        but internally permutes to (batch, seq_len, channels) for the transformer. Defaults match
        https://github.com/johahi/borzoi-pytorch/blob/main/borzoi_pytorch/config_borzoi.py defaults.
     -  we implement optional gradient checkpointing for memory efficiency during training
    
    Args:
        dim: Model dimension
        heads: Number of attention heads
        dim_key: Dimension of attention keys
        dim_value: Dimension of attention values
        attn_dropout: Dropout rate for attention
        pos_dropout: Dropout rate for positional encoding
        ffn_dropout: Dropout rate for feed-forward network
        num_rel_pos_features: Number of relative positional features
        use_ffn: Whether to include the feed-forward network (default: True)
        use_grad_checkpoint: Whether to use gradient checkpointing (default: False)
            Trades ~33% additional compute for significant memory savings. Only active
            during training. (default: False)
    """
    def __init__(
        self,
        dim,
        heads=8,
        dim_key=64,
        dim_value=192,
        attn_dropout=0.05,
        pos_dropout=0.01,
        ffn_dropout=0.2,
        num_rel_pos_features=32,
        use_ffn=True,
        flashed=False,
        use_grad_checkpoint=False,
    ):
        super().__init__()

        self.flashed = flashed
        self.use_grad_checkpoint = use_grad_checkpoint
        
        # Self-attention block with residual
        self.attn_block = Residual(nn.Sequential(
            nn.LayerNorm(dim, eps=0.001),
            Attention(
                dim=dim,
                heads=heads,
                dim_key=dim_key,
                dim_value=dim_value,
                dropout=attn_dropout,
                pos_dropout=pos_dropout,
                num_rel_pos_features=num_rel_pos_features
            ) if not self.flashed else
            FlashAttention(
                dim,
                heads = heads,
                dropout = attn_dropout,
                pos_dropout = pos_dropout,
            ),
            nn.Dropout(0.2)
        ))
        
        # Optional FFN block with residual
        if use_ffn:
            self.ffn_block = Residual(nn.Sequential(
                nn.LayerNorm(dim, eps=0.001),
                nn.Linear(dim, dim * 2),
                nn.Dropout(ffn_dropout),
                nn.ReLU(),
                nn.Linear(dim * 2, dim),
                nn.Dropout(ffn_dropout)
            ))
        else:
            self.ffn_block = None

    def _forward_impl(self, x):
        """
        Internal forward implementation that can be checkpointed.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels)
        
        Returns:
            Output tensor of shape (batch, seq_len, channels)
        """
        x = self.attn_block(x)
        if self.ffn_block is not None:
            x = self.ffn_block(x)
        return x
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
        
        Returns:
            Output tensor of shape (batch, channels, seq_len)
        """
        # Permute to transformer format: (B, C, L) → (B, L, C)
        x = x.permute(0, 2, 1)
        
        # Apply transformer blocks with optional checkpointing
        if self.use_grad_checkpoint and self.training:
            x = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            x = self._forward_impl(x)
        
        # Permute back to conv format: (B, L, C) → (B, C, L)
        x = x.permute(0, 2, 1)
        
        return x

@gin.configurable
@gin.register
class ConvDropout(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dropout, weight_decay=0, pad=False):
        super().__init__()
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
    def __init__(
        self, 
        in_channels, 
        filters, 
        pool_size=1, 
        pool_class=nn.AvgPool1d, 
        kernel_size=1, 
        shared_head=False, 
        stride=1, 
        weight_decay=0, 
        pad=False,
        init_weight=None,
        init_bias=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size
        self.pool = pool_class(pool_size)
        self.stride = stride
        self.shared_head = shared_head # this sets the output head for all the output tracks to be the same
        self.weight_decay = weight_decay
        if self.shared_head:
            self.conv = nn.Conv1d(in_channels, 1, kernel_size, stride=stride, padding=(kernel_size-1)//2 if pad else 0) # only one filter
        else:
            self.conv = nn.Conv1d(in_channels, filters, kernel_size, stride=stride, padding=(kernel_size-1)//2 if pad else 0) # multiple different output head filters
        if init_weight is not None:
            with torch.no_grad():
                self.conv.weight.fill_(init_weight)
        if init_bias is not None:
            with torch.no_grad():
                self.conv.bias.fill_(init_bias)
    def forward(self, x):
        x = self.conv(x)
        if self.shared_head:
            x = x.repeat(1, self.filters, 1) # duplicate output value across all tracks
        x = self.pool(x)
        return x

################################################################################################################
####                                        Layer Splitting Classes                                         ####
################################################################################################################

@gin.configurable
@gin.register
class ChannelSplitter(nn.Module):
    def __init__(self, split_ranges):
        super().__init__()
        self.split_ranges = split_ranges

    def forward(self,x):
        return tuple([x[:,start:end,:] for start,end in self.split_ranges])

@gin.configurable
@gin.register
class ChannelMerger(nn.Module):
    def forward(self,x_tuple):
        return torch.cat(x_tuple,dim=1)

class SplitModule(nn.Module):
    def _calculate_split_filters(self,filters,split_fracs):
        if sum(split_fracs)==1.0:
            split_filters = [round(frac*filters) for frac in split_fracs]
            diff = filters - sum(split_filters)
            if diff != 0:
                max_idx = split_filters.index(max(split_filters))
                split_filters[max_idx] += diff       
            return tuple(split_filters)
        else:
            raise ValueError(f"split_fracs {split_fracs} do not add to 1")   

@gin.configurable
@gin.register
class LearnedWeightedSum(nn.Module):
    def __init__(self, num_tensors):
        super().__init__()
        # Initialize learnable weights for each tensor in the tuple
        self.weights = nn.Parameter(torch.ones(num_tensors))

    def forward(self, x_tuple):
        # Normalize the weights (optional, but ensures they sum to 1)
        normalized_weights = torch.softmax(self.weights, dim=0)

        # Linearly combine the tensors in the tuple with learned weights
        weighted_sum = sum(normalized_weights[i] * x_tuple[i] for i in range(len(x_tuple)))
        
        return weighted_sum

@gin.configurable
@gin.register
class SplitConvDNA(SplitModule):
    def __init__(self, in_channels, filters, kernel_size, pool_size, input_splits, output_splits, weight_decay=0, pad=False):
        super().__init__()
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.weight_decay = weight_decay

        self.split_channels = self._calculate_split_filters(in_channels,input_splits)
        self.split_filters = self._calculate_split_filters(filters,output_splits)

        self.conv_dna_list = nn.ModuleList()
        for submodule_in_channels,submodule_filters in zip(self.split_channels,self.split_filters):
            self.conv_dna_list.append(
                ConvDNA(submodule_in_channels, submodule_filters, kernel_size, pool_size, weight_decay, pad)
            )
        

    def forward(self, x):
        output_tensors = []
        for tensor_idx, conv_dna in enumerate(self.conv_dna_list):
            output_tensors.append(conv_dna(x[tensor_idx]))
        return tuple(output_tensors)

@gin.configurable
@gin.register
class SplitConvTower(SplitModule):
    def __init__(self, in_channels, filters_init, filters_end, divisible_by, kernel_size, pool_size, repeat, input_splits, output_splits, weight_decay=0, pad=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.repeat = repeat
        self.weight_decay = weight_decay

        self.split_channels = self._calculate_split_filters(in_channels,input_splits)
        self.split_filters_init = self._calculate_split_filters(filters_init,output_splits)
        self.split_filters_end = self._calculate_split_filters(filters_end,output_splits)

        self.conv_tower_list = nn.ModuleList()
        for submodule_in_channels,submodule_filters_init,submodule_filters_end in zip(self.split_channels,self.split_filters_init,self.split_filters_end):
            self.conv_tower_list.append(
                ConvTower(submodule_in_channels, submodule_filters_init, submodule_filters_end, divisible_by, kernel_size, pool_size, repeat, weight_decay, pad)
            )

    def forward(self, x):
        output_tensors = []
        for tensor_idx, conv_tower in enumerate(self.conv_tower_list):
            output_tensors.append(conv_tower(x[tensor_idx]))
        return tuple(output_tensors)

@gin.configurable
@gin.register
class SplitConvBlock(SplitModule):
    def __init__(self, in_channels, filters, kernel_size, input_splits, output_splits, dilation=1, weight_decay=0, pad=False):
        super().__init__()
        self.kernel_size=kernel_size
        self.dilation=dilation
        self.weight_decay = weight_decay

        self.split_channels = self._calculate_split_filters(in_channels,input_splits)
        self.split_filters = self._calculate_split_filters(filters,output_splits)
        
        self.conv_list = nn.ModuleList()
        for submodule_in_channels,submodule_filters in zip(self.split_channels,self.split_filters):
            self.conv_list.append(
                ConvBlock(submodule_in_channels, submodule_filters, kernel_size, dilation, weight_decay, pad)
            )    

    def forward(self, x):
        output_tensors = []
        for tensor_idx, conv in enumerate(self.conv_list):
            output_tensors.append(conv(x[tensor_idx]))
        return tuple(output_tensors)

@gin.configurable
@gin.register
class SplitDilatedResidual(SplitModule):
    def __init__(
            self, 
            in_channels, 
            filters, 
            kernel_size, 
            rate_mult, 
            repeat, 
            input_splits,
            output_splits,
            dropout=0, 
            pad=False,
            ):
        super().__init__()
        self.kernel_size = kernel_size
        self.rate_mult = rate_mult
        self.repeat = repeat
        self.pad = pad

        self.split_channels = self._calculate_split_filters(in_channels,input_splits)
        self.split_filters = self._calculate_split_filters(filters,output_splits)

        self.dilated_residual_list = nn.ModuleList()
        for submodule_in_channels,submodule_filters in zip(self.split_channels,self.split_filters):
            self.dilated_residual_list.append(
                DilatedResidual(
                    submodule_in_channels, 
                    submodule_filters, 
                    kernel_size, 
                    rate_mult, 
                    repeat, 
                    dropout,
                    pad,
                )
            )

    def forward(self, x):
        output_tensors = []
        for tensor_idx, dilated_residual in enumerate(self.dilated_residual_list):
            output_tensors.append(dilated_residual(x[tensor_idx]))
        return tuple(output_tensors)

@gin.configurable
@gin.register
class SplitConvDropout(SplitModule):
    def __init__(self, in_channels, filters, kernel_size, dropout, input_splits, output_splits, weight_decay=0, pad=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay

        self.split_channels = self._calculate_split_filters(in_channels,input_splits)
        self.split_filters = self._calculate_split_filters(filters,output_splits)

        self.conv_dropout_list = nn.ModuleList()
        for submodule_in_channels,submodule_filters in zip(self.split_channels,self.split_filters):
            self.conv_dropout_list.append(
                ConvDropout(submodule_in_channels, submodule_filters, kernel_size, dropout, weight_decay, pad)
            )

    def forward(self, x):
        output_tensors = []
        for tensor_idx, conv_dropout in enumerate(self.conv_dropout_list):
            output_tensors.append(conv_dropout(x[tensor_idx]))
        return tuple(output_tensors)

@gin.configurable
@gin.register
class SplitConvFinal(SplitModule):
    def __init__(self, in_channels, filters, input_splits, output_splits, kernel_size=1, shared_head=False, stride=1, weight_decay=0, pad=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.shared_head = shared_head # this sets the output head for all the output tracks to be the same
        self.weight_decay = weight_decay

        self.split_channels = self._calculate_split_filters(in_channels,input_splits)
        self.split_filters = self._calculate_split_filters(filters,output_splits)

        self.conv_final_list = nn.ModuleList()
        for submodule_in_channels,submodule_filters in zip(self.split_channels,self.split_filters):
            self.conv_final_list.append(
                ConvFinal(submodule_in_channels, submodule_filters, kernel_size, shared_head, stride, weight_decay, pad)
            )
            
    def forward(self, x):
        output_tensors = []
        for tensor_idx, conv_final in enumerate(self.conv_final_list):
            output_tensors.append(conv_final(x[tensor_idx]))
        return tuple(output_tensors)