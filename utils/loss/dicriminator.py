import torch
from torch import nn
from utils.loss.utils import get_loss_on_result


def dis_loss(d_pred, d_target):
    sum_pred, losses_pred = get_loss_on_result(d_pred, 0)
    sum_tar, losses_tar = get_loss_on_result(d_target, 1)
    return sum_tar + sum_pred, losses_pred, losses_tar
