import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
from abc import ABC, abstractmethod
import random
from methylseqnet import tensor_ops

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
    def __call__(self, sequence, methylation, target, mask):
        """Apply the transform to the data."""
        pass

class LayerTransform(nn.Module):
    """
    Base class for all model layer transforms.

    These transforms are applied within the pytorch model, as part of the computational graph and on the GPU. Use
    for more computationally heavy-lift tasks or tasks that only impact inputs.
    """
    def __init__(self):
        super().__init__()
    def forward(self):
        raise NotImplementedError("Subclasses must implement this method")

################################################################################################################
####                                        LoaderTransform classes                                         ####
################################################################################################################

@gin.register
@gin.configurable
class CenteredSyntheticCpG(LoaderTransform):
    def __init__(self, window_size, center_cpg_frac, background_cpg_frac=None):
        """
        Apply a synthetic methylation landscape with one methylation fraction in a centered window 
        in the middle of the input sequence and another fraction along the rest of the sequence. 

        Returned input tensor will provide info for all CpG sites within the center window and 
        also in the background if background_cpg_frac is not None.

        Args:
            window_size: the size of the window, in bp, that will get center_cpg_frac. window_size//2 in each
                direction from seq_len//2
            center_cpg_frac: a fraction between 0 and 1 for how methylated CpGs in the window will be
            background_cpg_frac: fraction between 0 and 1 OR None. If None, background methylation landscape is
                left unchanged. If float, landscape at all background CpGs set to background_cpg_frac.
        """
        self.window_size = window_size
        self.center_cpg_frac = center_cpg_frac
        self.background_cpg_frac = background_cpg_frac
    def __call__(self, sequence, methylation, target, mask):
        """
        Args:
            sequence (torch.Tensor): Input tensor of shape (num_samples, num_variants, 4, seq_length) or
                unbatched tensor (num_variants, 4, seq_length)
            methylation (torch.Tensor): Input tensor of shape (num_samples, num_variants, num_cell_types, 3, seq_length) or
                unbatched tensor (num_variants, num_cell_types, 3, seq_length)
            target (torch.Tensor): Target tensor of shape (num_samples, num_tasks, track_length) or
                unbatched tensor (num_tasks, track_length)
            mask (torch.Tensor): Mask tensor of shape (num_samples, num_tasks, track_length) or
                unbatched tensor (num_tasks, track_length)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Transformed sequence, methylation, target, and mask.
        """
        methylation = methylation.clone()

        if methylation.dim()==4:
            for i in range(methylation.shape[0]):  # loop over samples
                for j in range(methylation.shape[1]):  # loop over variants
                    methylation[i,j] = self._apply_synthetic(methylation[i,j])
        elif methylation.dim()==3:
            for j in range(methylation.shape[1]):  # loop over variants
                methylation[j] = self._apply_synthetic(methylation[j])

        return input, target, mask

    def _apply_synthetic(self,seq):
        seq_len = seq.shape[-1]
        center = seq_len // 2
        half_window = self.window_size // 2
        window_start = max(center - half_window, 0)
        window_end = min(center + half_window, seq_len)

        # First apply center_cpg_frac to center window
        self._apply_synthetic_to_window(seq, (window_start, window_end), self.center_cpg_frac)

        # Then apply background_cpg_frac outside window if specified
        if self.background_cpg_frac is not None:
            if window_start > 0:
                self._apply_synthetic_to_window(seq, (0, window_start), self.background_cpg_frac)
            if window_end < seq_len:
                self._apply_synthetic_to_window(seq, (window_end, seq_len), self.background_cpg_frac)

        return seq
        
    def _apply_synthetic_to_window(self, seq, window, frac):
        """
        Apply a methylation fraction to input tensor seq, all methylation channels, within window (start,end)
        Args:
            seq (torch.Tensor): (num_channels, seq_length)
            window (Tuple[int,int]): (start, end) indices
            frac (float): fraction between 0 and 1
        """
        raise NotImplementedError("This is not implemented correctly currently; must fix handling of separate seq and methyl tensors")
        start, end = window
        channels = seq.shape[0]
        seq_len = seq.shape[-1]

        if (channels - 4) % 3 != 0:
            raise ValueError(
                f"Unexpected number of channels ({channels}); expected 4 + 3*N channels "
                "with ACGT first, followed by (mC, mG, CpG_indicator) triplets."
            )

        C_channel = 1
        G_channel = 2
    
        padded_start = max(start - 1, 0)
        padded_end = min(end + 1, seq_len)
    
        is_C_full = seq[C_channel, padded_start:padded_end] > 0.5
        is_G_full = seq[G_channel, padded_start:padded_end] > 0.5
    
        cpg_sites_full = is_C_full[:-1] & is_G_full[1:]
    
        cpg_mask_full = torch.zeros(padded_end - padded_start, dtype=torch.bool, device=seq.device)
        cpg_mask_full[:-1] |= cpg_sites_full  # mark 'C' position
        cpg_mask_full[1:]  |= cpg_sites_full  # mark 'G' position
    
        offset = start - padded_start  # offset to align window inside padded
        cpg_mask = cpg_mask_full[offset:offset + (end - start)]
        is_C = is_C_full[offset:offset + (end - start)]
        is_G = is_G_full[offset:offset + (end - start)]
    
        for base_channel in range(6, channels, 3):
            mC_channel = base_channel - 2
            mG_channel = base_channel - 1
            CpG_channel = base_channel
    
            seq[CpG_channel, start:end][cpg_mask] = 1.0
    
            seq[mC_channel, start:end][cpg_mask & is_C] = frac
            seq[mG_channel, start:end][cpg_mask & is_G] = frac
        
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

    def __call__(self, sequence, methylation, target, mask):
        """
        Args:
            sequence (torch.Tensor): Input tensor of shape (num_samples, num_variants, 4, seq_length) or
                unbatched tensor (num_variants, 4, seq_length)
            methylation (torch.Tensor): Input tensor of shape (num_samples, num_variants, num_cell_types, 3, seq_length) or
                unbatched tensor (num_variants, num_cell_types, 3, seq_length)
            target (torch.Tensor): Target tensor of shape (num_samples, num_tasks, track_length) or
                unbatched tensor (num_tasks, track_length)
            mask (torch.Tensor): Mask tensor of shape (num_samples, num_tasks, track_length) or
                unbatched tensor (num_tasks, track_length)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Transformed sequence, methylation, target, and mask.
        """
        if self.batch_wise or sequence.dim() == 3:
            if random.random() < self.probability:
                # reverse batch
                sequence = sequence.flip(dims=[-1])
                methylation = methylation.flip(dims=[-1])
                target = target.flip(dims=[-1])
                mask = mask.flip(dims=[-1])
                # complement batch
                sequence = self._apply_sequence_complement(sequence)
                methylation = self._apply_methylation_complement(methylation)

        else:
            for i in range(sequence.shape[0]):
                if random.random() < self.probability:
                    # reverse sample
                    sequence[i] = sequence[i].flip(dims=[-1])
                    methylation[i] = methylation[i].flip(dims=[-1])
                    target[i] = target[i].flip(dims=[-1])
                    mask[i] = mask[i].flip(dims=[-1])
                    # complement sample
                    sequence[i:i+1] = self._apply_sequence_complement(sequence[i:i+1]).clone()
                    methylation[i:i+1] = self._apply_methylation_complement(methylation[i:i+1]).clone()

        return sequence, methylation, target, mask

    def _apply_sequence_complement(self, sequence):
        """Swap A <-> T, C <-> G."""
        return sequence[..., [3, 2, 1, 0], :] # Swap: A<->T (0<->3), C<->G (1<->2)

    def _apply_methylation_complement(self, methylation):
        """Swap forward_strand_mC <-> reverse_strand_mC"""
        return methylation[..., [1, 0, 2], :] # Swap strands

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

    def __call__(self, sequence, methylation, target, mask):
        """
        Args:
            sequence (torch.Tensor): Input tensor of shape (num_samples, num_variants, 4, seq_length) or
                unbatched tensor (num_variants, 4, seq_length)
            methylation (torch.Tensor): Input tensor of shape (num_samples, num_variants, num_cell_types, 3, seq_length) or
                unbatched tensor (num_variants, num_cell_types, 3, seq_length)
            target (torch.Tensor): Target tensor of shape (num_samples, num_tasks, track_length) or
                unbatched tensor (num_tasks, track_length)
            mask (torch.Tensor): Mask tensor of shape (num_samples, num_tasks, track_length) or
                unbatched tensor (num_tasks, track_length)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Transformed sequence, methylation, target, and mask.
        """
        if self.batch_wise or sequence.dim() == 3:
            jitter_amount = self._draw_jitter()
            sequence = self._apply_jitter(sequence, jitter_amount)
            methylation = self._apply_jitter(methylation, jitter_amount)
        else:
            for i in range(sequence.shape[0]):
                jitter_amount = self._draw_jitter()
                sequence[i] = self._apply_jitter(sequence[i], jitter_amount)
                methylation[i] = self._apply_jitter(methylation[i], jitter_amount)

        return sequence, methylation, target, mask

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
class TrimOffEnds1d(LayerTransform):
    """
    TrimOffEnds1d will trim off the beginning and end of x along the sequence-length dimension and leave channels/sample untouched
    """
    def __init__(self,off_each_end):
        super().__init__()
        self.off_each_end=off_each_end
    def forward(self,x):
        if self.off_each_end: # if self.off_each_end if 0, None, or otherwise undefined, don't trim
            return x[:,:,self.off_each_end:-self.off_each_end]
        else:
            return x

@gin.configurable
@gin.register
class EncodingSelector(LayerTransform):
    """
    EncodingSelector will take a 7-dimensional input encoding ACGT-mCfrac-mGfrac-CpGmask and select a 
    different encoding for test purposes, such as seq-only, no CpGmask, or C + mCfrac add to 1.

    TODO: change to match:case statement
    """
    def __init__(self, encoding_str, window_size = 129):
        super().__init__()
        self.encoding_str = encoding_str
        self.window_size = window_size
        if self.encoding_str in [
            'seq+methyl_binary-seq',
            'seq+methyl_ACGTm-sum-to-1',
            'seq+methyl_binarize-methyl',
            'seq+interp-methyl+cpg-density',
        ]:
            self.channels = 7
        elif self.encoding_str in ['seq+methyl_no-mask','seq+interp-methyl']:
            self.channels = 6
        elif self.encoding_str in ['seq+methyl_combine-strands-no-mask','seq+smoothed-methyl']:
            self.channels = 5
        elif self.encoding_str in ['seq-only']:
            self.channels = 4
        elif self.encoding_str in ['stranded-methyl-with-mask']:
            self.channels = 3
        elif self.encoding_str in ['smoothed-methyl-only','interp-methyl-only']:
            self.channels = 1
        else:
            raise NotImplementedError(f"encoding_str: {self.encoding_str}")

    def forward(self, x):
        # the structure of the one-hot sequence coming in is [sample,(A,C,G,T,meth_fraction_fwd,meth_fraction_rev,valid_cpg),position]

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
        elif self.encoding_str == 'seq+interp-methyl+cpg-density':
            x = x.clone()
            x[:,4:6,:] = tensor_ops.interpolate_stranded_methylation(x)
            cpg_mask = x[:, 6, :] > 0
            x[:,6,:] = tensor_ops.tensor_rolling_average(cpg_mask,self.window_size)
        elif self.encoding_str == 'seq+methyl_combine-strands-no-mask':
            x = x.clone()
            x[:,4,:] = x[:,4,:] + x[:,5,:]
            x = x[:,0:5,:]
        elif self.encoding_str == 'seq+smoothed-methyl':
            mask = x[:,6,:]>0
            methylation = x[:,4,:]+x[:,5,:]
            smoothed = tensor_ops.mask_normalized_rolling_average(input=methylation,mask=mask,window_size=self.window_size)
            x = x.clone()
            x[:,4,:] = smoothed
            x = x[:,0:5,:]
        elif self.encoding_str == 'seq-only':
            x = x[:,0:4,:]
        elif self.encoding_str == 'stranded-methyl-with-mask':
            x = x[:,4:7,:]
        elif self.encoding_str == 'smoothed-methyl-only':
            mask = x[:,6,:]>0
            methylation = x[:,4,:]+x[:,5,:]
            smoothed = tensor_ops.mask_normalized_rolling_average(input=methylation,mask=mask,window_size=self.window_size)
            x = x.clone()
            x[:,4,:] = smoothed
            x = x[:,4:5,:]
        elif self.encoding_str == 'interp-methyl-only':
            x = x.clone()
            # for some reason tensor_ops.interpolate_collapsed_methylation isn't putting the tensor on the right device
            x[:,4:5,:] = tensor_ops.interpolate_collapsed_methylation(x)
            x = x[:,4:5,:]
        else:
            raise NotImplementedError(f"encoding_str: {self.encoding_str}")

        return x

@gin.configurable
@gin.register
class CpGSparsifier(LayerTransform):
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
        super().__init__()
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
        super().__init__()
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