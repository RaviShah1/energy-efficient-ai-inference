import torch
import torch.nn as nn
import torch.nn.functional as F


def get_ce_loss():
    return nn.CrossEntropyLoss()

def get_distillation_loss(temperature=4.0, alpha=0.25):
    ce_loss_fn = nn.CrossEntropyLoss()

    def loss_fn(student_logits, teacher_logits, labels):
        T = temperature
        soft_targets = F.softmax(teacher_logits / T, dim=1)
        soft_prob = F.log_softmax(student_logits / T, dim=1)

        # Soft targets loss (KL-like divergence)
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size(0)
        soft_targets_loss = soft_targets_loss * (T ** 2)

        # Hard label loss
        label_loss = ce_loss_fn(student_logits, labels)

        # Weighted combination
        loss = alpha * soft_targets_loss + (1. - alpha) * label_loss
        return loss

    return loss_fn