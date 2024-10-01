import torch
import torch.nn as nn
import torch.nn.functional as F
import gin

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