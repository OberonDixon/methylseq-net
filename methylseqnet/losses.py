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
            loss = (loss * mask.float()).sum() / (mask.float().sum() + self.eps)
        else:
            loss = loss.mean()
        logging.debug(f"{self.__class__.__name__} loss {loss}.")
        return loss

@gin.register
@gin.configurable
class BCELoss(MaskedLoss):
    def __init__(self,pos_weight=None,with_logits=False,eps=1e-7,**kwargs):
        super().__init__()
        self.eps=eps
        self.with_logits=with_logits
        if self.with_logits:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight,reduction='none',**kwargs)
        else:
            self.bce = nn.BCELoss(weight=pos_weight,reduction='none',**kwargs)
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
        poisson_weight: float = 1.0,
        multinomial_weight: float = 1.0,
        eps: float = 1e-7,
        log_input: bool = False,
        spatial: bool = False,
        subsetted: bool = False,
        subsets: list[list[int]] = None,
    ):
        super().__init__()
        self.poisson_weight = poisson_weight
        self.multinomial_weight = multinomial_weight
        self.eps = eps
        self.log_input = log_input
        self.spatial = spatial
        self.subsetted = subsetted
        self.subsets = subsets

    def forward(
        self,
        predictions: torch.Tensor,   # (N, C, L)
        targets: torch.Tensor,       # (N, C, L)
        mask: torch.Tensor = None,   # (N, C, L)
    ) -> torch.Tensor:
        if self.log_input:
            predictions = torch.exp(predictions)

        N, C, L = predictions.shape

        if mask is None:
            mask = torch.ones_like(predictions, dtype=torch.bool)
            logging.debug(f"{self.__class__.__name__} running without a mask; setting mask to all True.")
        mask_float = mask.float()

        # ---------- Poisson Term ----------
        if self.spatial:
            poisson_loss = F.poisson_nll_loss(
                predictions * mask_float,
                targets * mask_float,
                log_input=False,
                reduction='none',
            )  # (N, L)
            poisson_term = (poisson_loss * mask_float).sum() / (mask_float.sum() + self.eps)

        else:
            total_input = (predictions * mask_float).sum(dim=2)  # (N, C)
            total_target = (targets * mask_float).sum(dim=2)     # (N, C)

            poisson_loss = F.poisson_nll_loss(
                total_input,
                total_target,
                log_input=False,
                reduction='none',
            )  # (N, C)

            valid_sample_tracks = (mask_float.sum(dim=2) > 0).float()  # (N,)
            poisson_term = (poisson_loss * valid_sample_tracks).sum() / (valid_sample_tracks.sum() + self.eps)

        logging.debug(f"{self.__class__.__name__} poisson_term {poisson_term}; will be weighted by {self.poisson_weight}.")

        # ---------- Multinomial Term ----------
        if self.subsetted:
            if self.subsets:
                subsets = self.subsets
            else:
                raise ValueError(
                    f"{self.__class__.__name__} cannot be subsetted if subsets=None. "
                    "Please provide channel subsets upon which to perform taskwise multinomial loss."
                )
        else:
            subsets = [list(range(C))]
            
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
                pred_sum = pred_sub.sum(dim=1, keepdim=True) + self.eps  # (N, 1, L)
                log_p = torch.log(pred_sub / pred_sum + self.eps) # (N, 1, L)

                # Only compute loss where valid
                loss = -(targ_sub * log_p).sum(dim=1)  # (N, L)
                valid_pos = (mask_sub.sum(dim=1) > 0).float()  # (N, L)
                loss = (loss * valid_pos).sum() / (valid_pos.sum() + self.eps)  # scalar

            else:
                pred_sum = (pred_sub * mask_sub).sum(dim=2)  # (N, C')
                targ_sum = (targ_sub * mask_sub).sum(dim=2)  # (N, C')
                mask_sum = mask_sub.sum(dim=2)               # (N, C')

                total_pred = pred_sum.sum(dim=1, keepdim=True) + self.eps  # (N, 1)
                log_p = torch.log(pred_sum / total_pred + self.eps)  # (N, 1)

                valid = (mask_sum > 0).float()
                loss = -(targ_sum * log_p * valid).sum(dim=1)  # (N,)
                denom = valid.sum(dim=1) + self.eps
                loss = (loss / denom).mean()  # scalar

            multinomial_terms.append(loss)

        multinomial_term = torch.stack(multinomial_terms).mean()

        logging.debug(f"{self.__class__.__name__} multinomial_term {multinomial_term}; will be weighted by {self.multinomial_weight}.")

        return self.poisson_weight * poisson_term + self.multinomial_weight * multinomial_term