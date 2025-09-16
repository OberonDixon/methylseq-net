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
        if targets.ndim != 2 or predictions.ndim != 2:
            raise ValueError("Both targets and predictions must be 2D tensors of shape (channels, positions).")
        if targets.shape != predictions.shape:
            raise ValueError("Targets and predictions must have the same shape.")

class PearsonAcrossPositions(MultitaskMetric):
    def __call__(self, targets, predictions):
        self._check_shapes(targets, predictions)
        targets_centered = targets - targets.mean(dim=0, keepdim=True)  # center over channels
        predictions_centered = predictions - predictions.mean(dim=0, keepdim=True)

        numerator = (targets_centered * predictions_centered).sum(dim=0)
        denominator = torch.sqrt((targets_centered ** 2).sum(dim=0) * (predictions_centered ** 2).sum(dim=0))

        r = numerator / (denominator + 1e-8)
        return torch.mean(r)

class PearsonAcrossTasks(MultitaskMetric):
    def __call__(self, targets, predictions):
        self._check_shapes(targets, predictions)
        targets = targets.T
        predictions = predictions.T
        targets_centered = targets - targets.mean(dim=0, keepdim=True)  # center over positions
        predictions_centered = predictions - predictions.mean(dim=0, keepdim=True)

        numerator = (targets_centered * predictions_centered).sum(dim=0)
        denominator = torch.sqrt((targets_centered ** 2).sum(dim=0) * (predictions_centered ** 2).sum(dim=0))

        r = numerator / (denominator + 1e-8)
        return torch.mean(r)
        

