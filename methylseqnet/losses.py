import torch
import torch.nn as nn
import gin
import torch.nn.functional as F
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

class MaskedLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def forward(self,*args,mask=None):
        pass

@gin.register
class LogL1Loss(MaskedLoss):
    def __init__(self):
        super().__init__()
    def forward(self,activations,mask=None):
        return torch.clamp(activations, min=1e-8).log().abs().mean()

@gin.register
class LogL2Loss(MaskedLoss):
    def __init__(self):
        super().__init__()
    def forward(self,activations,mask=None):
        return (torch.clamp(activations, min=1e-8).log()**2).mean()

@gin.register
@gin.configurable
class PoissonLoss(MaskedLoss):
    def __init__(self,log_input=False,eps=1e-7,**kwargs):
        super().__init__()
        self.eps=eps
        self.poisson = nn.PoissonNLLLoss(log_input=log_input,reduction='none',**kwargs)
    def forward(self,predictions,targets,mask=None):
        loss = self.poisson(predictions,targets)
        if mask is not None:
            return (loss * mask.float()).sum() / (mask.float().sum() + self.eps)
        else:
            return loss.mean()

@gin.register
@gin.configurable
class BCELoss(MaskedLoss):
    def __init__(self,pos_weight=100,eps=1e-7,**kwargs):
        super().__init__()
        self.eps=eps
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight,reduction='none',**kwargs)
    def forward(self,predictions,targets,mask=None):
        loss = self.bce(predictions,targets)
        if mask is not None:
            return (loss * mask.float()).sum() / (mask.float().sum() + self.eps)
        else:
            loss = loss.mean()
            logging.debug(f"{self.__class__.__name__} running without a mask, calculated {loss.item()}")
            return loss.mean()

@gin.register
@gin.configurable
class PoissonMultinomialLoss(MaskedLoss):
    """
    Adapted from Avantika et al, 2025
    https://www.biorxiv.org/content/10.1101/2024.10.09.617507v3
    See Decima repo for original code:
    https://github.com/Genentech/decima/blob/main/src/decima/loss.py
    """
    def __init__(
        self,
        multinomial_weight: float = 1.0,
        eps: float = 1e-7,
        log_input: bool = False,
        spatial: bool = False,
        subsets: list[list[int]] = None,
    ):
        super().__init__()
        self.multinomial_weight = multinomial_weight
        self.eps = eps
        self.log_input = log_input
        self.spatial = spatial
        self.subsets = subsets

    def forward(
        self,
        predictions: torch.Tensor,        # (B, T, L)
        targets: torch.Tensor,       # (B, T, L)
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.log_input:
            predictions = torch.exp(predictions)

        B, T, L = predictions.shape

        if mask is None:
            mask = torch.ones_like(predictions, dtype=torch.bool)
            logging.debug(f"{self.__class__.__name__} running without a mask; setting mask to all True.")
        mask_float = mask.float()

        # ---------- Poisson Term ----------
        if self.spatial:
            total_input_per_pos = (predictions * mask_float).sum(dim=1)   # (B, L)
            total_target_per_pos = (targets * mask_float).sum(dim=1)      # (B, L)

            poisson_loss = F.poisson_nll_loss(
                total_input_per_pos,
                total_target_per_pos,
                log_input=False,
                reduction='none',
            )  # (B, L)

            valid_positions = (mask_float.sum(dim=1) > 0).float()  # (B, L)
            poisson_term = (poisson_loss * valid_positions).sum() / (valid_positions.sum() + self.eps)

        else:
            total_input = (predictions * mask_float).sum(dim=(1, 2))  # (B,)
            total_target = (targets * mask_float).sum(dim=(1, 2))     # (B,)

            poisson_loss = F.poisson_nll_loss(
                total_input,
                total_target,
                log_input=False,
                reduction='none',
            )  # (B,)

            valid_samples = (mask_float.sum(dim=(1, 2)) > 0).float()  # (B,)
            poisson_term = (poisson_loss * valid_samples).sum() / (valid_samples.sum() + self.eps)

        # ---------- Multinomial Term ----------
        subsets = self.subsets or [list(range(T))]
        multinomial_terms = []

        for subset in subsets:
            subset_tensor = torch.tensor(subset, device=predictions.device, dtype=torch.long)
            pred_sub = predictions[:, subset_tensor, :]
            targ_sub = targets[:, subset_tensor, :]
            mask_sub = mask[:, subset_tensor, :].float()

            if self.spatial:
                # Masked inputs and targets
                pred_sub = pred_sub * mask_sub
                targ_sub = targ_sub * mask_sub

                # Normalize across tasks at each position
                pred_sum = pred_sub.sum(dim=1, keepdim=True) + self.eps  # (B, 1, L)
                log_p = torch.log(pred_sub / pred_sum + self.eps)

                # Only compute loss where valid
                loss = -(targ_sub * log_p).sum(dim=1)  # (B, L)
                valid_pos = (mask_sub.sum(dim=1) > 0).float()  # (B, L)
                loss = (loss * valid_pos).sum() / (valid_pos.sum() + self.eps)

            else:
                pred_sum = (pred_sub * mask_sub).sum(dim=2)  # (B, T')
                targ_sum = (targ_sub * mask_sub).sum(dim=2)  # (B, T')
                mask_sum = mask_sub.sum(dim=2)               # (B, T')

                total_pred = pred_sum.sum(dim=1, keepdim=True) + self.eps  # (B, 1)
                log_p = torch.log(pred_sum / total_pred + self.eps)

                valid = (mask_sum > 0).float()
                loss = -(targ_sum * log_p * valid).sum(dim=1)  # (B,)
                denom = valid.sum(dim=1) + self.eps
                loss = (loss / denom).mean()  # scalar

            multinomial_terms.append(loss)

        multinomial_term = self.multinomial_weight * torch.stack(multinomial_terms).sum()

        return poisson_term + multinomial_term