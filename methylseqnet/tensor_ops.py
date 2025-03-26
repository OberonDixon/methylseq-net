import torch
import torch.nn.functional as F

def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int=-1, extrapolate: str='constant') -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Adapted from https://github.com/pytorch/pytorch/issues/50334 by GitHub user https://github.com/MoritzLange. No licensing provided.
    
    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)
    
    m = torch.diff(fp) / torch.diff(xp) # slope
    b = fp[..., :-1] - m * xp[..., :-1] # offset
    indices = torch.searchsorted(xp, x, right=False)
    
    if extrapolate == 'constant':
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else: # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)
    
    return values.movedim(-1, dim)

def interpolate_stranded_methylation(x: torch.Tensor) -> torch.Tensor:
    """
    Linearly interpolate methylation values across CpG sites on both strands,
    using only valid C or G sites (with CpG context) as anchors.
    
    x: [B, C, L] tensor with:
        - channel 1: C one-hot
        - channel 2: G one-hot
        - channel 4: forward strand methylation
        - channel 5: reverse strand methylation
        - channel 6: CpG site indicator

    Returns:
        x: same shape, with channels 4 and 5 replaced by interpolated values
    """
    B, C, L = x.shape
    all_idx = torch.arange(L, device=x.device).expand(B, L)

    # Build masks for valid C and G sites
    is_valid_c = (x[:, 1, :] > 0) & (x[:, 6, :] > 0)
    is_valid_g = (x[:, 2, :] > 0) & (x[:, 6, :] > 0)

    # Get counts per batch to pad correctly
    max_c = is_valid_c.sum(dim=1).max().item()
    max_g = is_valid_g.sum(dim=1).max().item()

    # Initialize padded xp/fp for both strands
    xp_c = torch.full((B, max_c), -1.0, device=x.device)
    fp_c = torch.zeros((B, max_c), device=x.device)

    xp_g = torch.full((B, max_g), -1.0, device=x.device)
    fp_g = torch.zeros((B, max_g), device=x.device)

    for b in range(B):
        idx_c = is_valid_c[b].nonzero(as_tuple=True)[0]
        idx_g = is_valid_g[b].nonzero(as_tuple=True)[0]

        xp_c[b, :len(idx_c)] = idx_c.float()
        fp_c[b, :len(idx_c)] = x[b, 4, idx_c]

        xp_g[b, :len(idx_g)] = idx_g.float()
        fp_g[b, :len(idx_g)] = x[b, 5, idx_g]

    # Interpolate both strands
    output_two_strands = torch.zeros((B,2,L))
    output_two_strands[:, 0, :] = interp(all_idx.float(), xp_c, fp_c, dim=-1, extrapolate='constant')
    output_two_strands[:, 1, :] = interp(all_idx.float(), xp_g, fp_g, dim=-1, extrapolate='constant')

    return output_two_strands

def interpolate_collapsed_methylation(x: torch.Tensor) -> torch.Tensor:
    """
    Collapse stranded methylation (channels 4 & 5) into a single signal.
    Sum both strands at CpG positions (channel 6 > 0), and interpolate across sequence.

    Args:
        x: Tensor of shape [B, C, L], where:
            - Channel 4: forward methylation
            - Channel 5: reverse methylation
            - Channel 6: CpG mask

    Returns:
        Tensor of shape [B, 1, L] with interpolated, combined methylation
    """
    B, C, L = x.shape
    all_idx = torch.arange(L, device=x.device).expand(B, L)

    methyl_mask = x[:, 6, :] > 0
    methyl_sum = (x[:, 4, :] + x[:, 5, :]) * methyl_mask

    max_points = methyl_mask.sum(dim=1).max().item()
    xp = torch.full((B, max_points), -1.0, device=x.device)
    fp = torch.zeros((B, max_points), device=x.device)

    for b in range(B):
        idx = methyl_mask[b].nonzero(as_tuple=True)[0]
        xp[b, :len(idx)] = idx.float()
        fp[b, :len(idx)] = methyl_sum[b, idx]

    interpolated = interp(all_idx.float(), xp, fp, dim=-1, extrapolate='constant')

    output_one_strand = torch.zeros((B,1,L))
    output_one_strand[:, 0, :] = interpolated

    return output_one_strand

def tensor_rolling_average(input: torch.Tensor, window_size: int) -> torch.Tensor:
    kernel = torch.ones(1, 1, window_size, device=input.device)
    padded = F.pad(input.unsqueeze(1), (window_size//2, window_size//2), mode='constant', value=0)
    density = F.conv1d(padded.float(), kernel) / window_size
    return density.squeeze(1)

def mask_normalized_rolling_average(input: torch.Tensor, mask: torch.Tensor, window_size: int) -> torch.Tensor:
    padding = (window_size - 1) // 2
    kernel = torch.ones(1, 1, window_size, device=input.device)  # Create a kernel with ones
    smoothed = F.conv1d(input.unsqueeze(1), kernel, padding=padding).squeeze(1)  # Apply convolution
    mask_sum = F.conv1d(mask.unsqueeze(1).float(), kernel, padding=padding).squeeze(1)
    smoothed = torch.nan_to_num((smoothed / mask_sum),nan=1,posinf=1)
    return smoothed
    