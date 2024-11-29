import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
from abc import ABC, abstractmethod
import random

# TODO: move over CpGSparsifier, EncodingSelector. Rename SmoothMethylation too. Keep old versions for now; obsolete at a later point

"""
This module is going to end up containing LoaderTransforms and LayerTransforms, the former occuring *within the dataset loading operation* and applying to *inputs and outputs* while the latter occur within *model forward pass* and operate *only on input data*
"""

################################################################################################################
####                                         Abstract Base Classes                                          ####
################################################################################################################

class LoaderTransform(ABC):
    """
    Base class for all loader transforms.

    These transforms are applied by the dataloader on the CPU. Use for low-parallelization / low compute overhead
    tasks or tasks that must operate on both inputs and targets.
    """
    
    @abstractmethod
    def __call__(self, input, target, mask):
        """Apply the transform to the data."""
        pass

class LayerTransform(nn.Module):
    """
    Base class for all model layer transforms.

    These transforms are applied within the pytorch model, as part of the computational graph and on the GPU. Use
    for more computationally heavy-lift tasks or tasks that only impact inputs.
    """

    def forward(self):
        raise NotImplementedError("Subclasses must implement this method")

################################################################################################################
####                                        LoaderTransform classes                                         ####
################################################################################################################

@gin.register
@gin.configurable
class ReverseComplement(LoaderTransform):
    def __init__(self, probability=0.5, batch_wise=True):
        """
        Apply reverse complement with a given probability.
        Args:
            probability (float): Probability of applying reverse complement.
            batch_wise (bool): If True, applies the same reverse complement across the whole batch.
                               If False, applies reverse complement on a per-sample basis.
        """
        self.probability = probability
        self.batch_wise = batch_wise

    def __call__(self, input, target, mask):
        """
        Args:
            input (torch.Tensor): Input tensor of shape (num_samples, num_channels, seq_length).
            target (torch.Tensor): Target tensor of shape (sample, task_idx, position).
            mask (torch.Tensor): Mask tensor of shape (sample, task_idx, position).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Transformed input, target, and mask.
        """
        if self.batch_wise:
            if random.random() < self.probability:
                # reverse batch
                input = input.flip(dims=[-1])
                target = target.flip(dims=[-1])
                mask = mask.flip(dims=[-1])
                # complement batch
                input = self._apply_complement(input)

        else:
            for i in range(input.shape[0]):
                if random.random() < self.probability:
                    # reverse sample
                    input[i] = input[i].flip(dims=[-1])
                    target[i] = target[i].flip(dims=[-1])
                    mask[i] = mask[i].flip(dims=[-1])
                    # complement sample
                    input[i:i+1,:,:] = self._apply_complement(input[i:i+1,:,:])

        return input, target, mask

    def _apply_complement(self, seq):
        """Swap A <-> T, C <-> G, and forward_strand_mC <-> reverse_strand_mC in the input sequence."""
        seq = seq.clone()
        seq[:,[0, 3],:] = seq[:,[3, 0],:]  # Swap onehotA with onehotT
        seq[:,[1, 2],:] = seq[:,[2, 1],:]  # Swap onehotC with onehotG
        seq[:,[4, 5],:] = seq[:,[5, 4],:]  # Swap methylationfracfwd with methylationfracrev
        return seq

@gin.register
@gin.configurable
class SequenceJitter(LoaderTransform):
    def __init__(self, stdev=10, max_jitter=32, batch_wise=True):
        """
        Apply jitter by shifting the sequence and target randomly forward or backward.
        Args:
            max_jitter (int): Maximum amount of jitter (base pairs).
            batch_wise (bool): If True, applies the same jitter across the whole batch.
                               If False, applies jitter on a per-sample basis.
        """
        self.stdev = stdev
        self.max_jitter = max_jitter
        self.batch_wise = batch_wise

    def __call__(self, input, target, mask):
        """
        Args:
            input (torch.Tensor): Input tensor of shape (num_samples, num_channels, seq_length).
            target (torch.Tensor): Target tensor of shape (sample, task_idx, position).
            mask (torch.Tensor): Mask tensor of shape (sample, task_idx, position).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Jittered input, target, and mask.
        """
        if self.batch_wise:
            jitter_amount = self._draw_jitter()
            input = self._apply_jitter(input, jitter_amount)
        else:
            for i in range(input.shape[0]):
                jitter_amount = self._draw_jitter()
                input[i] = self._apply_jitter(input[i], jitter_amount)

        return input, target, mask

    def _draw_jitter(self):
        """
        Draws a jitter amount from a normal distribution, clamped by the max_jitter.
        Returns:
            jitter_amount (int): Jitter amount (positive or negative), clamped to max_jitter.
        """
        # Draw jitter from a normal distribution with mean 0 and standard deviation `self.stdev`
        jitter_amount = torch.normal(mean=0.0, std=self.stdev, size=(1,)).item()
        
        # Clamp the jitter to be within the range [-max_jitter, max_jitter]
        jitter_amount = int(torch.clamp(torch.tensor(jitter_amount), -self.max_jitter, self.max_jitter).item())
        
        return jitter_amount
    
    def _apply_jitter(self, seq, jitter_amount):
        """Shift the sequence forward or backward and pad with zeros."""
        seq = seq.clone()
        if jitter_amount > 0:
            seq = torch.cat([torch.zeros(seq.shape[:-1] + (jitter_amount,)), seq[..., :-jitter_amount]], dim=-1)
        elif jitter_amount < 0:
            seq = torch.cat([seq[..., -jitter_amount:], torch.zeros(seq.shape[:-1] + (-jitter_amount,))], dim=-1)
        return seq

################################################################################################################
####                                         LayerTransform classes                                         ####
################################################################################################################

@gin.configurable
@gin.register
class CpGSparsifier(nn.Module):
    """
    Randomly add sparsity to the CpG methylation input tracks, if available. By default this is only done at training
    time.
    Args:
        in_channels: the number of encoding channels coming in. Channels 0,1,2,3 are assumed to be sequence channels.
        remove_fracs: different fractional removal options. If more than one is provided, each tensor will have a random
            fraction removed, e.g. if remove_fracs=[0,0.2,0.99], one third of the time 99% of CpG information will be
            removed and one third of the time no CpG information will be removed. The same fraction is applied to the 
            whole batch, but specific CpGs are selected one sample at a time.
        chunk_size: CpGs will be kept or removed in chunks of this size. Using chunk_size=1 will mean that most removed
            methylation information can be "filled in" from adjacent CpGs
        training_only: if training_only=True (default), sparsification will only occur at training time and not eval/test
    """
    def __init__(
        self, 
        in_channels, 
        remove_fracs=[0.2], 
        chunk_size=8, 
        training_only=True,
    ):
        super(CpGSparsifier, self).__init__()
        self.in_channels=in_channels
        self.remove_fracs=remove_fracs
        self.chunk_size=chunk_size
        self.training_only=training_only
    def forward(self,x):
        if self.in_channels>4 and (not self.training_only or self.training):
            fraction = random.choice(self.remove_fracs)
            num_samples, num_channels, length = x.shape
            effective_length = (length + self.chunk_size - 1) // self.chunk_size
            chunk_mask = torch.rand(num_samples, effective_length, device=x.device) > fraction
            mask = chunk_mask.repeat_interleave(self.chunk_size, dim=1)
            mask = mask[:,:length]
            mask = mask.unsqueeze(1).expand(num_samples, self.in_channels-4, length)
            x[:,4:,:]*=mask # Zero out the methylation channels using the mask
        return x

################################################################################################################
####                                          Obsolete Old Classes                                          ####
################################################################################################################

class SmoothMethylationTransform(nn.Module):
    def __init__(self, window_size=3):
        """
        Initializes the smoothing transform.
        Args:
            window_size (int): Size of the smoothing window. Should be odd to ensure a symmetric window.
        """
        super(SmoothMethylationTransform,self).__init__()
        self.window_size = window_size

    def smooth(self, methylation, mask):
        """
        Apply smoothing to methylation data using a rolling window, but only on positions where mask == 1.
        
        Args:
            methylation (torch.Tensor): Tensor of methylation data (batch_size, position).
            mask (torch.Tensor): Binary tensor (batch_size, position), where 1 indicates valid positions for smoothing.
        
        Returns:
            torch.Tensor: Smoothed methylation tensor.
        """
        # Apply padding to allow windowed operation
        padding = (self.window_size - 1) // 2

        # Convolution for smoothing with a uniform kernel
        kernel = torch.ones(1, 1, self.window_size, device=methylation.device)  # Create a kernel with ones
        smoothed = F.conv1d(methylation.unsqueeze(1), kernel, padding=padding).squeeze(1)  # Apply convolution

        # Normalize by the sum of valid positions in the window (mask)
        mask_sum = F.conv1d(mask.unsqueeze(1).float(), kernel, padding=padding).squeeze(1)
        mask_sum = mask_sum.clamp(min=1)  # Avoid division by zero

        # Apply the mask (keep smoothed values only where mask == 1)
        smoothed = (smoothed / mask_sum) * mask

        return smoothed

    def forward(self, x):
        """
        Args:
            input_data (torch.Tensor): Tensor with shape (batch_size, channels, position).
                                       Channels 5 and 6 are methylation values;
                                       Channels 0-3 are sequence one-hot encoded (A, C, G, T);
                                       Channel 7 is CG mask.

        Returns:
            torch.Tensor: Transformed input_data with smoothed methylation values.
        """
        # Get channels 5, 6 (methylation), 7 (mask), and 0-3 (sequence)
        methylation_forward = x[:, 4, :]
        methylation_reverse = x[:, 5, :]
        sequence_A = x[:, 0, :]
        sequence_C = x[:, 1, :]
        sequence_G = x[:, 2, :]
        sequence_T = x[:, 3, :]
        cg_mask = x[:, 6, :]  # CG positions mask

        # Create forward and reverse strand masks
        forward_strand_mask = (sequence_C > 0) & (cg_mask > 0)  # C must be present in the sequence
        reverse_strand_mask = (sequence_G > 0) & (cg_mask > 0)  # G must be present in the sequence

        # Smooth forward strand methylation
        smoothed_forward = self.smooth(methylation_forward, forward_strand_mask)

        # Smooth reverse strand methylation
        smoothed_reverse = self.smooth(methylation_reverse, reverse_strand_mask)

        # Update the input data tensor with smoothed methylation
        x[:, 4, :] = smoothed_forward
        x[:, 5, :] = smoothed_reverse

        return x