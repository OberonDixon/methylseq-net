from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc, f1_score
import torch

def all_metrics(targets,probabilities):
    predictions = (probabilities >= 0.5).astype('int')
    
    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(targets, predictions)
    precision_overall = precision_score(targets, predictions,zero_division=0)
    recall_overall = recall_score(targets, predictions,zero_division=0)
    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    # Calculate the area under the precision-recall curve
    pr_auc = auc(recall, precision)
    f1 = f1_score(targets, predictions, average='binary')
    
    return {
            "accuracy":accuracy,
            "precision":precision_overall,
            "recall":recall_overall,
            "f1":f1,
            "prc_auc":pr_auc,
        }

class MultitaskMetric(ABC):
    @abstractmethod
    def __call__(self, targets, predictions):
        """
        Take in two 3D tensors of shape (num_variants, channels, positions) and return a scalar tensor.
        Both tensors must have the same shape.
        """
        pass
    def _check_shapes(self, targets, predictions):
        if targets.ndim != 3 or predictions.ndim != 3:
            raise ValueError("Both targets and predictions must be 3D tensors of shape (num_variants, channels, positions).")
        if targets.shape != predictions.shape:
            raise ValueError("Targets and predictions must have the same shape.")

class PearsonAcrossPositions(MultitaskMetric):
    """
    Computes Pearson correlation across positions for each variant×channel pair, then returns the mean.
    
    Filtering: Positions with counts ≤ min_counts are NaN-masked within each variant×channel.
    Returns: Mean correlation across all variant×channel pairs, or NaN if positions=1 or no active data.
    """
    def __init__(self, min_counts=5):
        self.min_counts = min_counts
    def __call__(self, targets, predictions):
        """
        Take in two 3D tensors of shape (num_variants, channels, positions) and return a scalar tensor.
        Both tensors must have the same shape.
        """
        self._check_shapes(targets, predictions)

        num_variants, channels, positions = targets.shape

        targets = targets.reshape(num_variants * channels, positions)
        predictions = predictions.reshape(num_variants * channels, positions)

        active_mask = (targets > self.min_counts)

        if positions == 1 or active_mask.sum() == 0:
            # Cannot compute Pearson correlation with no active positions
            return torch.tensor(float('nan'),device=targets.device)

        targets_nanmasked = torch.where(active_mask, targets, torch.nan)
        predictions_nanmasked = torch.where(active_mask, predictions, torch.nan)

        targets_centered = targets_nanmasked - torch.nanmean(targets_nanmasked, dim=1, keepdim=True)  # center over positions
        predictions_centered = predictions_nanmasked - torch.nanmean(predictions_nanmasked, dim=1, keepdim=True)

        numerator = torch.nansum(targets_centered * predictions_centered, dim=1)
        denominator = torch.sqrt(torch.nansum((targets_centered ** 2),dim=1) * torch.nansum((predictions_centered ** 2), dim=1))

        r = numerator / (denominator + 1e-8)
        return torch.nanmean(r)

class PearsonAcrossTasks(MultitaskMetric):
    """
    Computes Pearson correlation across channels for each variant×position pair, then returns 
    variance-weighted mean.
    
    Filtering: Drops variant×position pairs where all channels have counts ≤ min_counts.
    Scaling: Correlations weighted by target variance at each variant×position.
    Returns: Variance-weighted mean, or NaN if channels=1 or no active data.
    """
    def __init__(self, min_counts=5):
        self.min_counts = min_counts
    def __call__(self, targets, predictions):
        """
        Take in two 3D tensors of shape (num_variants, channels, positions) and return a scalar tensor.
        Both tensors must have the same shape.
        """
        self._check_shapes(targets, predictions)

        num_variants, channels, positions = targets.shape

        targets = targets.permute(1,0,2).reshape(channels, num_variants * positions)
        predictions = predictions.permute(1,0,2).reshape(channels, num_variants * positions)

        active_mask = (targets > self.min_counts).any(dim=0)

        if channels == 1 or active_mask.sum() == 0:
            # Cannot compute Pearson correlation with only one channel or not active positions
            return torch.tensor(float('nan'),device=targets.device)

        targets = targets[:, active_mask]
        predictions = predictions[:, active_mask]

        targets_centered = targets - targets.mean(dim=0, keepdim=True)  # center over positions
        predictions_centered = predictions - predictions.mean(dim=0, keepdim=True)
        position_variance = targets.var(dim=0)

        numerator = (targets_centered * predictions_centered).sum(dim=0)
        denominator = torch.sqrt((targets_centered ** 2).sum(dim=0) * (predictions_centered ** 2).sum(dim=0))

        r = numerator / (denominator + 1e-8)
        return (r * position_variance).sum() / (position_variance.sum() + 1e-8)     

class CCCAcrossVariants(MultitaskMetric):
    """
    Computes concordance correlation coefficient (CCC) across variants for each channel×position 
    feature, then returns the mean.
    
    Filtering: Drops channel×position features where all variants have counts < min_counts.
    Returns: Mean CCC across all features, or NaN if num_variants=1 or no active data.
    """
    def __init__(self, min_counts=5):
        self.min_counts = min_counts
    
    def __call__(self, targets, predictions):
        """
        Take in two 3D tensors of shape (num_variants, channels, positions) and return a scalar tensor.
        Both tensors must have the same shape.
        """
        self._check_shapes(targets, predictions)

        num_variants, channels, positions = targets.shape

        targets = targets.reshape(num_variants, channels * positions)
        predictions = predictions.reshape(num_variants, channels * positions)

        active_mask = (targets >= self.min_counts).any(dim=0)

        if num_variants == 1 or active_mask.sum() == 0:
            # Cannot compute CCC with only one variant or no active positions
            return torch.tensor(float('nan'),device=targets.device)

        targets = targets[:, active_mask]
        predictions = predictions[:, active_mask]

        mean_targets = targets.mean(dim=0, keepdim=True)
        mean_predictions = predictions.mean(dim=0, keepdim=True)
        
        targets_centered = targets - mean_targets
        predictions_centered = predictions - mean_predictions
        
        # Compute Pearson correlation coefficient ρ for each channel×position
        numerator_pearson = (targets_centered * predictions_centered).mean(dim=0)
        std_targets = torch.sqrt((targets_centered ** 2).mean(dim=0))
        std_predictions = torch.sqrt((predictions_centered ** 2).mean(dim=0))
        rho = numerator_pearson / (std_targets * std_predictions + 1e-8)
        
        # Compute CCC: ρc = 2ρσxσy / (σx² + σy² + (μx - μy)²)
        var_targets = std_targets ** 2
        var_predictions = std_predictions ** 2
        mean_diff_sq = (mean_targets.squeeze() - mean_predictions.squeeze()) ** 2
        
        ccc = (2 * rho * std_targets * std_predictions) / (var_targets + var_predictions + mean_diff_sq + 1e-8)
        
        return ccc.mean()