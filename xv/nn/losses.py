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
        self.reduction = reduction

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

class FocalLossMulticlass(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



class TorchFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, targets):
        max_val = (-outputs).clamp(min=0)
        log_ = ((-max_val).exp() + (-outputs - max_val).exp()).log()
        loss = outputs - outputs * targets + max_val + log_

        invprobs = F.logsigmoid(-outputs * (targets * 2.0 - 1.0))
        loss = self.alpha*(invprobs * self.gamma).exp() * loss
        return loss.mean()

loss_dict = {
    'bcewithlogits': nn.BCEWithLogitsLoss,
    'focal': TorchFocalLoss,
    'jaccard': JaccardLoss,
    'dice': DiceLoss,
    'focalmulti': FocalLossMulticlass,
}
