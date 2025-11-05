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
        Take in two 2D tensors of shape (channels, positions) and return a scalar value.
        Both tensors must have the same shape.
        """
        pass
    def _check_shapes(self, targets, predictions):
        if targets.ndim != 3 or predictions.ndim != 3:
            raise ValueError("Both targets and predictions must be 3D tensors of shape (num_variants, channels, positions).")
        if targets.shape != predictions.shape:
            raise ValueError("Targets and predictions must have the same shape.")

class PearsonAcrossPositions(MultitaskMetric):
    def __init__(self, min_counts=5):
        self.min_counts = min_counts
    def __call__(self, targets, predictions):
        self._check_shapes(targets, predictions)

        num_variants, channels, positions = targets.shape

        if positions == 1:
            return torch.tensor(float('nan'))  # Cannot compute Pearson correlation with only one position

        targets = targets.reshape(num_variants * channels, positions)
        predictions = predictions.reshape(num_variants * channels, positions)

        active_mask = (targets > self.min_counts)
        targets_nanmasked = torch.where(active_mask, targets, torch.nan)
        predictions_nanmasked = torch.where(active_mask, predictions, torch.nan)

        targets_centered = targets_nanmasked - torch.nanmean(targets_nanmasked, dim=1, keepdim=True)  # center over channels
        predictions_centered = predictions_nanmasked - torch.nanmean(predictions_nanmasked, dim=1, keepdim=True)

        numerator = torch.nansum(targets_centered * predictions_centered, dim=1)
        denominator = torch.sqrt(torch.nansum((targets_centered ** 2),dim=1) * torch.nansum((predictions_centered ** 2), dim=1))

        r = numerator / (denominator + 1e-8)
        return torch.nanmean(r)

class PearsonAcrossTasks(MultitaskMetric):
    def __init__(self, min_count=5):
        self.min_count = min_count
    def __call__(self, targets, predictions):
        self._check_shapes(targets, predictions)

        num_variants, channels, positions = targets.shape

        if channels == 1:
            return torch.tensor(float('nan'))  # Cannot compute Pearson correlation with only one channel

        targets = targets.reshape(channels, num_variants * positions)
        predictions = predictions.reshape(channels, num_variants * positions)

        active_mask = (targets > self.min_count).any(dim=0)
        targets = targets[:, active_mask]
        predictions = predictions[:, active_mask]

        targets_centered = targets - targets.mean(dim=0, keepdim=True)  # center over positions
        predictions_centered = predictions - predictions.mean(dim=0, keepdim=True)
        position_variance = targets.var(dim=0)

        numerator = (targets_centered * predictions_centered).sum(dim=0)
        denominator = torch.sqrt((targets_centered ** 2).sum(dim=0) * (predictions_centered ** 2).sum(dim=0))

        r = numerator / (denominator + 1e-8)
        return (r * position_variance).sum() / (position_variance.sum() + 1e-8) if active_mask.sum() > 0 else torch.tensor(0.0)
        

