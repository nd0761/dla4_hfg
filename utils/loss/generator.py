import torch
from torch import nn
from utils.loss.utils import get_loss_on_result


def gen_loss(d_out):
    return get_loss_on_result(d_out, 1)
