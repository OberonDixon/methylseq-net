from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc, f1_score
import torch

def pearson_per_task(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation for each task (row).
    
    Args:
        predictions: (tasks, positions)
        targets: (tasks, positions)
    
    Returns:
        Tensor of shape (tasks,) with per-task Pearson r values.
    """
    # Center
    p = predictions - predictions.mean(dim=1, keepdim=True)
    t = targets - targets.mean(dim=1, keepdim=True)
    
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p ** 2).sum(dim=1) * (t ** 2).sum(dim=1))
    
    return num / den.clamp(min=1e-8)


def delta_z(r1, r2, eps=1e-7):
    z1 = torch.arctanh(torch.as_tensor(r1).clamp(-1 + eps, 1 - eps))
    z2 = torch.arctanh(torch.as_tensor(r2).clamp(-1 + eps, 1 - eps))
    return z1 - z2

def fisher_mean(correlations: torch.Tensor) -> float:
    """Average correlations via Fisher z-transformation.
    
    Args:
        correlations: 1-D tensor of Pearson r values.
    
    Returns:
        Back-transformed mean correlation (scalar).
    """
    z = torch.arctanh(correlations.clamp(-1 + 1e-7, 1 - 1e-7))
    return torch.tanh(z.mean()).item()

class GenomicTensorMetric(ABC):
    """
    Base class for multitask metrics operating on 3D tensors of shape
    (num_variants, channels, positions).

    Subclasses must implement the stateful update/compute/reset interface.
    The __call__ method is provided as a convenience for one-shot evaluation
    (e.g. per-sample metrics), and should not be used during epoch-level
    accumulation.
    """

    @abstractmethod
    def update(self, targets: torch.Tensor, predictions: torch.Tensor) -> None:
        """Accumulate sufficient statistics from one batch."""
        pass

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Return the final scalar metric from accumulated statistics."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state."""
        pass

    @abstractmethod
    def merge(self, other: "GenomicTensorMetric") -> None:
        """
        Fold another (possibly never-updated) instance's accumulated sufficient statistics
        into this one, as if this instance had also processed every batch accumulated into
        `other`. Used to pool per-rank accumulated state under DDP so compute() can run once
        on the true combined-epoch statistics, instead of averaging each rank's independently
        -computed local result (which is not the same statistic when ranks see unequal data).
        """
        pass

    def __call__(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        One-shot evaluation on a single tensor pair. Resets any existing state,
        accumulates the given tensors, computes the result, then resets again
        to leave the instance clean.
        """
        self.reset()
        self.update(targets, predictions)
        result = self.compute()
        self.reset()
        return result

    def _reshape_inputs(self, targets: torch.Tensor, predictions: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        if targets.ndim not in (3,4) or predictions.ndim not in (3,4):
            raise ValueError(
                "Both targets and predictions must be 3D (num_variants, channels, positions) "
                "or 4D (batch, num_variants, channels, positions) tensors."
            )
        if targets.shape != predictions.shape:
            raise ValueError("Targets and predictions must have the same shape.")
        if targets.ndim == 4:
            batch, num_variants, channels, positions = targets.shape
            targets = targets.permute(1, 2, 0, 3).reshape(num_variants, channels, batch * positions)
            predictions = predictions.permute(1, 2, 0, 3).reshape(num_variants, channels, batch * positions)
        return targets, predictions

class PearsonAcrossPositions(GenomicTensorMetric):
    """
    Computes Pearson correlation across positions for each variant×channel pair, then returns the mean.
    
    Filtering: Positions with counts ≤ min_counts are excluded from sufficient statistics.
    Returns: Mean correlation across all variant×channel pairs with at least one active position,
             or NaN if no active data exists across all accumulated batches.

    Sufficient statistics accumulated per variant×channel pair (shape: (num_variants*channels,)):
        n:      number of active positions
        sum_x:  sum of active target values
        sum_y:  sum of active prediction values
        sum_xx: sum of squared active target values
        sum_yy: sum of squared active prediction values
        sum_xy: sum of active target*prediction products
    """
    def __init__(self, min_counts=5):
        self.min_counts = min_counts
        self.reset()

    def reset(self):
        self.n = None
        self.sum_x = None
        self.sum_y = None
        self.sum_xx = None
        self.sum_yy = None
        self.sum_xy = None

    def merge(self, other: "PearsonAcrossPositions") -> None:
        if other is None or other.n is None:
            return
        if self.n is None:
            self.n      = other.n.clone()
            self.sum_x  = other.sum_x.clone()
            self.sum_y  = other.sum_y.clone()
            self.sum_xx = other.sum_xx.clone()
            self.sum_yy = other.sum_yy.clone()
            self.sum_xy = other.sum_xy.clone()
        else:
            self.n      = self.n      + other.n
            self.sum_x  = self.sum_x  + other.sum_x
            self.sum_y  = self.sum_y  + other.sum_y
            self.sum_xx = self.sum_xx + other.sum_xx
            self.sum_yy = self.sum_yy + other.sum_yy
            self.sum_xy = self.sum_xy + other.sum_xy

    def update(self, targets, predictions):
        """
        targets, predictions: (num_variants, channels, positions)
        Accumulates sufficient statistics, masking positions where targets <= min_counts.
        Shape of accumulated stats: (num_variants * channels,)
        """
        targets, predictions = self._reshape_inputs(targets, predictions)
        t = targets.float().reshape(-1, targets.shape[-1])   # (num_variants*channels, positions)
        p = predictions.float().reshape(-1, targets.shape[-1])
        mask = t > self.min_counts                           # (num_variants*channels, positions)

        zeros = torch.zeros_like(t)
        n_new      = mask.float().sum(dim=-1)
        sum_x_new  = torch.where(mask, t,     zeros).sum(dim=-1)
        sum_y_new  = torch.where(mask, p,     zeros).sum(dim=-1)
        sum_xx_new = torch.where(mask, t * t, zeros).sum(dim=-1)
        sum_yy_new = torch.where(mask, p * p, zeros).sum(dim=-1)
        sum_xy_new = torch.where(mask, t * p, zeros).sum(dim=-1)

        if self.n is None:
            self.n      = n_new
            self.sum_x  = sum_x_new
            self.sum_y  = sum_y_new
            self.sum_xx = sum_xx_new
            self.sum_yy = sum_yy_new
            self.sum_xy = sum_xy_new
        else:
            # accumulated stats may have different leading dim if channels differ across
            # dataset_keys — caller is responsible for not mixing keys into one instance
            self.n      = self.n      + n_new
            self.sum_x  = self.sum_x  + sum_x_new
            self.sum_y  = self.sum_y  + sum_y_new
            self.sum_xx = self.sum_xx + sum_xx_new
            self.sum_yy = self.sum_yy + sum_yy_new
            self.sum_xy = self.sum_xy + sum_xy_new

    def compute(self):
        """
        Returns mean Pearson r across all variant×channel pairs that have at least one
        active position. Pairs with n <= 1 are excluded (correlation undefined).
        Returns NaN if no valid pairs exist.
        """
        if self.n is None:
            return torch.tensor(float('nan'))

        valid = self.n > 1   # need at least 2 points for correlation to be defined
        if not valid.any():
            return torch.tensor(float('nan'))

        n      = self.n[valid]
        mean_x = self.sum_x[valid]  / n
        mean_y = self.sum_y[valid]  / n
        var_x  = (self.sum_xx[valid] / n - mean_x ** 2).clamp(min=0)
        var_y  = (self.sum_yy[valid] / n - mean_y ** 2).clamp(min=0)
        cov    = self.sum_xy[valid]  / n - mean_x * mean_y

        r = cov / (torch.sqrt(var_x * var_y) + 1e-8)
        return r.mean()

class PearsonAcrossTasks(GenomicTensorMetric):
    """
    Computes Pearson correlation across channels (tasks) for each variant×position pair,
    then returns a variance-weighted mean.

    Filtering: variant×position pairs where all channels have counts ≤ min_counts are dropped.
    Weighting: each pair's correlation is weighted by the target variance across channels.
    Returns: variance-weighted mean correlation, or NaN if channels ≤ 1 or no active data.

    Incremental strategy:
        At each position we can compute r_p (Pearson across C channels) and var_p
        (target variance across channels) entirely from the current batch — no need
        to remember per-position state across batches.  We then accumulate just two
        running scalars:
            weighted_r_sum  += sum_over_positions(r_p * var_p)
            var_sum         += sum_over_positions(var_p)
        Final result = weighted_r_sum / var_sum.

    Memory: O(channels × positions) per batch — never grows with dataset size.
    """

    def __init__(self, min_counts=5):
        self.min_counts = min_counts
        self.reset()

    def reset(self):
        self.weighted_r_sum = 0.0
        self.var_sum = 0.0

    def merge(self, other: "PearsonAcrossTasks") -> None:
        if other is None:
            return
        self.weighted_r_sum += other.weighted_r_sum
        self.var_sum += other.var_sum

    def update(self, targets, predictions):
        """
        targets, predictions: (num_variants, channels, positions)
        Computes per-position cross-channel Pearson r and target variance,
        then accumulates the variance-weighted sum.
        """
        targets, predictions = self._reshape_inputs(targets, predictions)
        num_variants, channels, positions = targets.shape

        if channels <= 1:
            return

        # Reshape to (channels, num_variants * positions) so each column is one
        # variant×position pair and correlation runs across channels (rows)
        t = targets.float().permute(1, 0, 2).reshape(channels, -1)  # (C, V*P)
        p = predictions.float().permute(1, 0, 2).reshape(channels, -1)

        # Filter: keep only positions where at least one channel > min_counts
        active = (t > self.min_counts).any(dim=0)  # (V*P,)
        if not active.any():
            return

        t = t[:, active]  # (C, n_active)
        p = p[:, active]

        # Pearson r across channels (dim=0) at each active position
        t_mean = t.mean(dim=0, keepdim=True)  # (1, n_active)
        p_mean = p.mean(dim=0, keepdim=True)
        t_c = t - t_mean
        p_c = p - p_mean

        cov = (t_c * p_c).sum(dim=0)                          # (n_active,)
        std_t = torch.sqrt((t_c ** 2).sum(dim=0))
        std_p = torch.sqrt((p_c ** 2).sum(dim=0))
        r = cov / (std_t * std_p + 1e-8)                      # (n_active,)

        # Target variance across channels at each position
        var_t = t.var(dim=0)                                   # (n_active,)

        # Accumulate
        self.weighted_r_sum += (r * var_t).sum().item()
        self.var_sum += var_t.sum().item()

    def compute(self):
        if self.var_sum < 1e-12:
            return torch.tensor(float('nan'))
        return torch.tensor(self.weighted_r_sum / self.var_sum) 

class CCCAcrossVariants(GenomicTensorMetric):
    """
    Computes concordance correlation coefficient (CCC) across variants for each channel×position
    feature, then returns the mean.

    Filtering: channel×position features where all variants have counts < min_counts are excluded.
    Returns: mean CCC across all active features, or NaN if num_variants=1 or no active data.

    Sufficient statistics accumulated per channel×position feature (shape: (channels*positions,)),
    treating the population of variants as the "positions" dimension:
        n:      total number of variants seen (scalar, same for all features after filtering)
        sum_x:  sum of target values across variants
        sum_y:  sum of prediction values across variants
        sum_xx: sum of squared target values across variants
        sum_yy: sum of squared prediction values across variants
        sum_xy: sum of target*prediction products across variants

    Note: unlike PearsonAcrossTasks, the active mask here is per-feature (any variant active),
    applied consistently across all features — inactive features accumulate zeros and are
    excluded in compute() rather than being filtered batch-by-batch. This is necessary because
    a feature inactive in one batch may be active in another.
    """
    def __init__(self, min_counts=5):
        self.min_counts = min_counts
        self.reset()

    def reset(self):
        self.n      = 0
        self.sum_x  = None
        self.sum_y  = None
        self.sum_xx = None
        self.sum_yy = None
        self.sum_xy = None
        self.any_active = None   # tracks which features were active in at least one batch

    def merge(self, other: "CCCAcrossVariants") -> None:
        if other is None or other.sum_x is None:
            return
        self.n = self.n + other.n
        if self.sum_x is None:
            self.sum_x      = other.sum_x.clone()
            self.sum_y      = other.sum_y.clone()
            self.sum_xx     = other.sum_xx.clone()
            self.sum_yy     = other.sum_yy.clone()
            self.sum_xy     = other.sum_xy.clone()
            self.any_active = other.any_active.clone()
        else:
            self.sum_x      = self.sum_x  + other.sum_x
            self.sum_y      = self.sum_y  + other.sum_y
            self.sum_xx     = self.sum_xx + other.sum_xx
            self.sum_yy     = self.sum_yy + other.sum_yy
            self.sum_xy     = self.sum_xy + other.sum_xy
            self.any_active = self.any_active | other.any_active

    def update(self, targets, predictions):
        """
        targets, predictions: (num_variants, channels, positions)
        Reshapes to (num_variants, channels*positions), accumulates sufficient statistics
        over the variants dimension per channel×position feature.
        """
        targets, predictions = self._reshape_inputs(targets, predictions)
        num_variants, channels, positions = targets.shape

        t = targets.float().reshape(num_variants, -1)    # (num_variants, channels*positions)
        p = predictions.float().reshape(num_variants, -1)

        batch_active = (t >= self.min_counts).any(dim=0)  # (channels*positions,)

        # accumulate which features have ever been active
        if self.any_active is None:
            self.any_active = batch_active
        else:
            self.any_active = self.any_active | batch_active

        # accumulate stats over all features (including inactive ones) —
        # inactive features will be masked out in compute() via any_active
        self.n = self.n + num_variants

        sum_x_new  = t.sum(dim=0)          # (channels*positions,)
        sum_y_new  = p.sum(dim=0)
        sum_xx_new = (t * t).sum(dim=0)
        sum_yy_new = (p * p).sum(dim=0)
        sum_xy_new = (t * p).sum(dim=0)

        if self.sum_x is None:
            self.sum_x  = sum_x_new
            self.sum_y  = sum_y_new
            self.sum_xx = sum_xx_new
            self.sum_yy = sum_yy_new
            self.sum_xy = sum_xy_new
        else:
            self.sum_x  = self.sum_x  + sum_x_new
            self.sum_y  = self.sum_y  + sum_y_new
            self.sum_xx = self.sum_xx + sum_xx_new
            self.sum_yy = self.sum_yy + sum_yy_new
            self.sum_xy = self.sum_xy + sum_xy_new

    def compute(self):
        """
        Returns mean CCC across all channel×position features that were active in at
        least one batch. Returns NaN if num_variants=1 or no active features exist.
        """
        if self.sum_x is None or self.n <= 1:
            return torch.tensor(float('nan'))
        if self.any_active is None or not self.any_active.any():
            return torch.tensor(float('nan'))

        # restrict to features active in at least one batch
        s_x  = self.sum_x[self.any_active]
        s_y  = self.sum_y[self.any_active]
        s_xx = self.sum_xx[self.any_active]
        s_yy = self.sum_yy[self.any_active]
        s_xy = self.sum_xy[self.any_active]

        n      = float(self.n)
        mean_x = s_x  / n
        mean_y = s_y  / n
        var_x  = (s_xx / n - mean_x ** 2).clamp(min=0)
        var_y  = (s_yy / n - mean_y ** 2).clamp(min=0)
        cov    = s_xy  / n - mean_x * mean_y

        # CCC = 2*cov / (var_x + var_y + (mean_x - mean_y)^2)
        mean_diff_sq = (mean_x - mean_y) ** 2
        ccc = (2 * cov) / (var_x + var_y + mean_diff_sq + 1e-8)

        return ccc.mean()