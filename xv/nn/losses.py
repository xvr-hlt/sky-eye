import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch import nn

class WeightedLoss(nn.Module):
    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
    
    def forward(self, outputs, targets):
        l = 0.
        for loss, weight in self.loss_weights.items():
            l += loss(outputs, targets)*weight
        return l/sum(self.loss_weights.values())
    
class JaccardLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, outputs, targets):
        axes = tuple(i for i, _ in enumerate(targets.shape))
        outputs = outputs.sigmoid()
        intersection = (outputs * targets).sum(axes[1:])
        cardinality = outputs.sum(axes[1:]) + targets.sum(axes[1:])
        jaccard_score = ((intersection + self.eps)/(cardinality - intersection + self.eps))
        target_none_mask = (targets.sum(axes[1:]) > 0).float()
        return ((1. - jaccard_score)*target_none_mask).mean()

class DiceLoss(JaccardLoss):
    def __init__(self, eps=1e-6):
        super().__init__(eps=eps)

    def forward(self, outputs, targets):
        jaccard = 1-super().forward(outputs, targets)
        return 1-(2 * jaccard/(1+jaccard))

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.0, alpha = 0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction="mean"

    def forward(self, outputs, targets):
        targets = targets.type(outputs.type())

        logpt = -F.binary_cross_entropy_with_logits(
            outputs, targets, reduction="none"
        )
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt).pow(self.gamma)) * logpt

        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1 - self.alpha) * (1 - targets))

        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        if self.reduction == "batchwise_mean":
            loss = loss.sum(0)

        return loss

def dice_round(preds, trues):
    preds = preds.float()
    return soft_dice_loss(preds, trues)


loss_dict = {
    'bcewithlogits': nn.BCEWithLogitsLoss,
    'focal': FocalLoss,
    'jaccard': JaccardLoss,
    'dice': DiceLoss,
}
