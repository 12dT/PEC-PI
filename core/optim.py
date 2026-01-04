"""
Optimizer and learning rate scheduler to replace dassl.optim
"""
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler
import math


def build_optimizer(model, optim_cfg):
    """Build optimizer from config."""
    optim_name = optim_cfg.NAME.lower()
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    
    # Get parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optim_name == "sgd":
        optimizer = SGD(
            params,
            lr=lr,
            momentum=optim_cfg.MOMENTUM,
            weight_decay=weight_decay,
            dampening=optim_cfg.SGD_DAMPNING,
            nesterov=optim_cfg.SGD_NESTEROV
        )
    elif optim_name == "adam":
        optimizer = Adam(
            params,
            lr=lr,
            betas=(optim_cfg.ADAM_BETA1, optim_cfg.ADAM_BETA2),
            weight_decay=weight_decay
        )
    elif optim_name == "adamw":
        optimizer = AdamW(
            params,
            lr=lr,
            betas=(optim_cfg.ADAM_BETA1, optim_cfg.ADAM_BETA2),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")
    
    return optimizer


def build_optimizer_from_params(param_groups, optim_cfg):
    """Build optimizer from parameter groups (for separate LRs)."""
    optim_name = optim_cfg.NAME.lower()
    weight_decay = optim_cfg.WEIGHT_DECAY
    
    if optim_name == "sgd":
        optimizer = SGD(
            param_groups,
            momentum=optim_cfg.MOMENTUM,
            weight_decay=weight_decay,
            dampening=optim_cfg.SGD_DAMPNING,
            nesterov=optim_cfg.SGD_NESTEROV
        )
    elif optim_name == "adam":
        optimizer = Adam(
            param_groups,
            betas=(optim_cfg.ADAM_BETA1, optim_cfg.ADAM_BETA2),
            weight_decay=weight_decay
        )
    elif optim_name == "adamw":
        optimizer = AdamW(
            param_groups,
            betas=(optim_cfg.ADAM_BETA1, optim_cfg.ADAM_BETA2),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")
    
    return optimizer


class WarmupScheduler(_LRScheduler):
    """Learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epoch, max_epoch, warmup_type="linear", 
                 warmup_cons_lr=1e-5, base_scheduler=None, last_epoch=-1):
        self.warmup_epoch = warmup_epoch
        self.max_epoch = max_epoch
        self.warmup_type = warmup_type
        self.warmup_cons_lr = warmup_cons_lr
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            # Warmup phase
            if self.warmup_type == "constant":
                return [self.warmup_cons_lr for _ in self.base_lrs]
            elif self.warmup_type == "linear":
                alpha = self.last_epoch / self.warmup_epoch
                return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Regular scheduling
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            return self.base_lrs
    
    def step(self, epoch=None):
        super().step(epoch)
        if self.last_epoch >= self.warmup_epoch and self.base_scheduler is not None:
            self.base_scheduler.step()


def build_lr_scheduler(optimizer, optim_cfg):
    """Build learning rate scheduler from config."""
    scheduler_name = optim_cfg.LR_SCHEDULER.lower()
    max_epoch = optim_cfg.MAX_EPOCH
    warmup_epoch = optim_cfg.WARMUP_EPOCH
    
    if scheduler_name == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epoch - warmup_epoch
        )
    elif scheduler_name == "step":
        step_size = optim_cfg.STEPSIZE[0] if optim_cfg.STEPSIZE else 10
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=optim_cfg.GAMMA
        )
    elif scheduler_name == "multistep":
        milestones = optim_cfg.STEPSIZE
        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=optim_cfg.GAMMA
        )
    else:
        # No scheduling
        base_scheduler = None
    
    if warmup_epoch > 0:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epoch=warmup_epoch,
            max_epoch=max_epoch,
            warmup_type=optim_cfg.WARMUP_TYPE,
            warmup_cons_lr=optim_cfg.WARMUP_CONS_LR,
            base_scheduler=base_scheduler
        )
    else:
        scheduler = base_scheduler
    
    return scheduler


def compute_accuracy(output, target, topk=(1,)):
    """Compute accuracy for classification."""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res


def compute_f1_score(all_preds, all_labels, num_classes=3):
    """
    Compute F1 score for multi-class classification
    
    Args:
        all_preds: predicted labels (numpy array or list)
        all_labels: ground truth labels (numpy array or list)
        num_classes: number of classes
    
    Returns:
        f1_macro: macro-averaged F1 score
        f1_per_class: F1 score for each class
    """
    from sklearn.metrics import f1_score, classification_report
    import numpy as np
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Macro F1 (average of per-class F1)
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
    
    # Weighted F1 (weighted by class frequency)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
    
    # Per-class F1
    f1_per_class = f1_score(all_labels, all_preds, average=None) * 100
    
    return f1_macro, f1_weighted, f1_per_class
