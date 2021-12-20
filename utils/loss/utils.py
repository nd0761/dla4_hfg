import torch


def get_loss_on_result(d_out, true_answer):
    losses_dis = []
    for d in d_out:
        temp = torch.mean((true_answer - d) ** 2)
        losses_dis.append(temp)
    return sum(losses_dis), losses_dis
