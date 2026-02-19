import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
from abc import ABC, abstractmethod
import random
from methylseqnet import tensor_ops
import warnings
import numpy as np

from methylseqnet.motifs import dinuc_shuffle

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
class IdentityTransform(LoaderTransform):
    def __call__(self, sequence, methylation, target, mask):
        return sequence, methylation, target, mask

@gin.register
@gin.configurable
class ZerosTransform(LoaderTransform):
    def __call__(self, sequence, methylation, target, mask):
        return torch.zeros_like(sequence), torch.zeros_like(methylation), torch.zeros_like(target), torch.zeros_like(mask)

class LoaderCpGTransform(LoaderTransform):
    """Base class for transforms that modify methylation at CpG sites."""
    def _get_cpg_masks(self, sequence):
        """
        Identify CpG dinucleotides from sequence tensor, returning separate masks for C and G positions.
        
        Args:
            sequence: Tensor of shape (..., 4, seq_length) where dim -2 is one-hot ACGT
        
        Returns:
            Tuple of (cpg_c_mask, cpg_g_mask):
                cpg_c_mask: Boolean tensor marking C positions in CpG sites
                cpg_g_mask: Boolean tensor marking G positions in CpG sites
        """
        C_channel = 1
        G_channel = 2
        
        # Get C and G positions
        is_C = sequence[..., C_channel, :] > 0.5  # (..., seq_length)
        is_G = sequence[..., G_channel, :] > 0.5  # (..., seq_length)
        
        # Check for CpG dinucleotides (C followed by G)
        is_cpg = is_C[..., :-1] & is_G[..., 1:]  # (..., seq_length-1)
        
        # Create separate masks for C and G positions
        seq_len = sequence.shape[-1]
        cpg_c_mask = torch.zeros_like(is_C, dtype=torch.bool)
        cpg_g_mask = torch.zeros_like(is_C, dtype=torch.bool)
        
        # Mark C positions (index i where CpG starts)
        cpg_c_mask[..., :-1] = is_cpg
        # Mark G positions (index i+1 where CpG continues)
        cpg_g_mask[..., 1:] = is_cpg
        
        return cpg_c_mask, cpg_g_mask

    def _apply_methylation(self, methylation, c_mask, g_mask, frac):
        """
        Apply methylation fraction and valid_cpg mask in-place to C and G positions indicated by masks.
        
        Args:
            methylation: Tensor of shape (..., num_cell_types, 3, seq_length)
                         where dim -2 has channels [mC, mG, CpG_indicator]
            c_mask: Boolean tensor of shape (..., seq_length) indicating C positions to modify
            g_mask: Boolean tensor of shape (..., seq_length) indicating G positions to modify
            frac: Methylation fraction to apply (0 to 1)
        """
        if not isinstance(frac, torch.Tensor):
            frac = torch.tensor(frac, device=methylation.device, dtype=methylation.dtype)
        else:
            frac = frac.to(device=methylation.device, dtype=methylation.dtype)
        # Expand masks to match methylation dimensions
        while c_mask.ndim < methylation.ndim - 1:
            c_mask = c_mask.unsqueeze(-2)
            g_mask = g_mask.unsqueeze(-2)
        
        # Set CpG indicator channel (index 2) to 1.0 for both C and G positions
        combined_mask = (c_mask | g_mask).expand_as(methylation[..., 2, :])
        methylation[..., 2, :] = torch.where(
            combined_mask,
            torch.tensor(1.0, device=methylation.device, dtype=methylation.dtype),
            methylation[..., 2, :]
        )
        
        # Set mC channel (index 0) to frac ONLY at C positions
        c_mask_expanded = c_mask.expand_as(methylation[..., 0, :])
        methylation[..., 0, :] = torch.where(
            c_mask_expanded,
            frac,
            methylation[..., 0, :]
        )
        
        # Set mG channel (index 1) to frac ONLY at G positions
        g_mask_expanded = g_mask.expand_as(methylation[..., 1, :])
        methylation[..., 1, :] = torch.where(
            g_mask_expanded,
            frac,
            methylation[..., 1, :]
        )

@gin.register
@gin.configurable
class InsertSyntheticCpG(LoaderCpGTransform):
    def __init__(
        self,
        center_window_size=500,
        flank_width=1000,
        offset=0,
        center_cpg_frac=None,
        flanking_cpg_frac=None,
        background_cpg_frac=None,
        ):
        """
        Apply a synthetic methylation landscape with specified methylation fractions at every CpG position
        in each of the following locations:
         - center window (center_window_size bp centered at seq_len//2 + offset)
         - flanking regions (flank_width bp on each side of center window)
         - background (all other CpGs outside center window and flanking regions)


        The sequence tensor is left untouched and only used to identify CpG locations. The methylation tensor
        is cloned and modified for each region according to the specified fractions, leaving the existing landscape
        unchanged where fractions are None.

        Args:
            center_window_size (int): Size of the center window in base pairs.
            flank_width (int): Width of flanking regions on each side of the center window.
            offset (int): Offset to apply to the center window position.
            center_cpg_frac (float or None): Methylation fraction to apply at CpGs in the center window.
            flanking_cpg_frac (float or None): Methylation fraction to apply at CpGs in the flanking regions.
            background_cpg_frac (float or None): Methylation fraction to apply at CpGs outside center and flanking regions.
        """
        self.window_size = center_window_size
        self.flank_width = flank_width
        self.offset = offset
        if center_cpg_frac is None and flanking_cpg_frac is None and background_cpg_frac is None:
            warnings.warn("center_cpg_frac, flanking_cpg_frac, and background_cpg_frac are all None: InsertSyntheticCpG will have no effect.")
        self.center_cpg_frac = center_cpg_frac
        self.flanking_cpg_frac = flanking_cpg_frac
        self.background_cpg_frac = background_cpg_frac
    def __call__(self, sequence, methylation, target, mask):
        """
        Args:
            sequence (torch.Tensor): Input tensor of shape (num_samples, num_variants, 4, seq_length) or
                unbatched tensor (num_variants, 4, seq_length)
            methylation (torch.Tensor): Input tensor of shape (num_samples, num_variants, num_cell_types, 3, seq_length) or
                unbatched tensor (num_variants, num_cell_types, 3, seq_length)
            target (torch.Tensor): Target tensor of shape (num_samples, num_tasks, track_length) or
                unbatched tensor (num_variants, num_tasks, track_length)
            mask (torch.Tensor): Mask tensor of shape (num_samples, num_tasks, track_length) or
                unbatched tensor (num_variants, num_tasks, track_length)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Transformed sequence, methylation, target, and mask.
        """
        seq_len = sequence.shape[-1]
        min_seq_len = self.window_size + 2 * self.flank_width + 2 * abs(self.offset)
        
        if min_seq_len > seq_len:
            warnings.warn(
                f"Center window size + 2*flank_width + 2 * abs(offset) ({min_seq_len}) "
                f"exceeds sequence length ({seq_len})."
            )
        
        methylation = methylation.clone()
        
        # Identify CpG sites from sequence (vectorized)
        cpg_c_mask, cpg_g_mask = self._get_cpg_masks(sequence)  # Shape matches sequence batch dims + (seq_len,)
        
        # Create region masks
        if self.window_size == -1:  # Special case: if window_size is -1, treat the whole sequence as the center window
            center_start = 0
            center_end = seq_len
        else:
            center = seq_len // 2 + self.offset
            half_window = self.window_size // 2
            # Center window
            center_start = max(center - half_window, 0)
            center_end = min(center + half_window, seq_len)
        
        # Flanking regions
        left_flank_start = max(center_start - self.flank_width, 0)
        left_flank_end = center_start
        right_flank_start = center_end
        right_flank_end = min(center_end + self.flank_width, seq_len)
        
        # Create position mask for each region
        positions = torch.arange(seq_len, device=sequence.device)
        
        center_mask = (positions >= center_start) & (positions < center_end)
        left_flank_mask = (positions >= left_flank_start) & (positions < left_flank_end)
        right_flank_mask = (positions >= right_flank_start) & (positions < right_flank_end)
        flank_mask = left_flank_mask | right_flank_mask
        background_mask = ~(center_mask | flank_mask)
        
        # Apply methylation fractions to each region
        if self.background_cpg_frac is not None:
            self._apply_methylation(methylation, cpg_c_mask & background_mask, 
                                   cpg_g_mask & background_mask, self.background_cpg_frac)
        
        if self.flanking_cpg_frac is not None:
            self._apply_methylation(methylation, cpg_c_mask & flank_mask, 
                                   cpg_g_mask & flank_mask, self.flanking_cpg_frac)
        
        if self.center_cpg_frac is not None:
            self._apply_methylation(methylation, cpg_c_mask & center_mask, 
                                   cpg_g_mask & center_mask, self.center_cpg_frac)
        
        return sequence, methylation, target, mask

@gin.register
@gin.configurable
class DinucShufflePreserveMethylation(LoaderCpGTransform):
    def __init__(self):
        """
        Dinucleotide-shuffle the sequence and remap the methylation landscape
        to the new CpG positions via interpolation.
        
        Used as an attribution baseline: destroys sequence motifs while
        preserving the regional methylation landscape, isolating sequence
        contributions in integrated gradients.
        """
        self.interp_methyl_from_encoding = EncodingSelector(encoding_str='interp-methyl-only')

    def __call__(self, sequence, methylation, target, mask):
        shuffled_sequence = torch.from_numpy(
            np.array(
                [
                    dinuc_shuffle(sequence.cpu().numpy()[b].transpose()).transpose()
                    for b in range(sequence.shape[0])
                ]
            )
        ).to(device=sequence.device, dtype=torch.float32) # Shape: (batch_size, 4, seq_length)
        cpg_c_mask, cpg_g_mask = self._get_cpg_masks(shuffled_sequence)
        remapped_methylations_list = []
        for state_idx in range(methylation.shape[-3]):
            interpolated_methylation = self.interp_methyl_from_encoding(torch.cat([sequence, methylation[...,state_idx,:,:]], dim=1)) # Shape: (batch_size, 1, seq_length)
            remapped_methylation = torch.zeros_like(methylation[...,state_idx,:,:]) # Shape: (batch_size, 3, seq_length)
            frac = interpolated_methylation.squeeze(1)
            self._apply_methylation(remapped_methylation, cpg_c_mask, cpg_g_mask, frac)
            remapped_methylations_list.append(remapped_methylation.unsqueeze(-3))
        remapped_methylation = torch.cat(remapped_methylations_list, dim=-3)
        return shuffled_sequence, remapped_methylation, target, mask

@gin.register
@gin.configurable
class DinucShuffleSyntheticCpG(LoaderCpGTransform):
    def __init__(self, cpg_frac):
        """
        Dinucleotide-shuffle the sequence and apply a uniform methylation fraction at all CpG sites
        in the shuffled sequence. Returns the shuffled sequence and modified methylation tensor.

        Used as an attribution baseline transform: the shuffled sequence preserves dinucleotide
        frequencies while destroying higher-order patterns, and the methylation is set to a
        neutral/modal state at the resulting CpG positions.

        Args:
            cpg_frac (float): Methylation fraction to apply at CpG sites (e.g., 0.95 for modal).
        """
        self.cpg_frac = cpg_frac
    def __call__(self, sequence, methylation, target, mask):
        shuffled_sequence = torch.from_numpy(
            np.array(
                [
                    dinuc_shuffle(sequence.cpu().numpy()[b].transpose()).transpose()
                    for b in range(sequence.shape[0])
                ]
            )
        ).to(device=sequence.device, dtype=torch.float32)
        remapped_methylation = torch.zeros_like(methylation)
        cpg_c_mask, cpg_g_mask = self._get_cpg_masks(shuffled_sequence)
        self._apply_methylation(remapped_methylation, cpg_c_mask, cpg_g_mask, self.cpg_frac)
        return shuffled_sequence, remapped_methylation, target, mask
        
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
        elif self.encoding_str in ['smoothed-methyl-only','interp-methyl-only', 'cpg-density-only']:
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
        elif self.encoding_str == 'cpg-density-only':
            cpg_mask = x[:, 6, :] > 0
            cpg_density = tensor_ops.tensor_rolling_average(cpg_mask,self.window_size)
            x = x.clone()
            x[:,6,:] = cpg_density
            x = x[:,6:7,:]
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

@gin.configurable
@gin.register
class SmoothChannels(nn.Module):
    def __init__(self, window_size=3, channel_indices=(4,5)):
        """
        Initializes the smoothing transform.
        Args:
            window_size (int): Size of the smoothing window. Should be odd to ensure a symmetric window.
            channel_indices (tuple of ints): Indices of the channel to smooth (default is (4,5),
            which corresponds to forward + rev strand methylation in standard input encoding).
        """
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size should be odd to ensure a symmetric window.")
        self.window_size = window_size
        self.channel_indices = channel_indices
        # Convolution for smoothing with a uniform kernel
        self.register_buffer('kernel', torch.ones(len(self.channel_indices), 1, self.window_size) / self.window_size)
        self.padding = (self.window_size - 1) // 2

    def forward(self, x):
        """
        Apply smoothing to the set of channels.
        
        Returns:
            torch.Tensor: tensor with appropriate channels smoothed.
        """
        x = x.clone()  # Avoid in-place modification
        smoothed = F.conv1d(x[:, self.channel_indices, :], self.kernel, padding=self.padding, groups=len(self.channel_indices))
        x[:,self.channel_indices,:] = smoothed

        return x