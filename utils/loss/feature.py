import torch
from torch import nn
from utils.loss.utils import get_loss_on_result


def feat_loss(f_pred, f_target):
    loss = 0
    for d_p, d_t in zip(f_pred, f_target):
        for d_1, d_2 in zip(d_p, d_t):
            loss += 2 * torch.mean(torch.abs(d_1 - d_2))

    return loss
