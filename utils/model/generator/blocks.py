import torch.nn as nn
from utils.config import TaskConfig

# source shw5


def get_padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):

    def __init__(self, hidden_channel, kr, Drs):
        super(ResBlock, self).__init__()

        self.net = []
        for i in range(len(Drs)):
            d12 = Drs[i]
            for d in d12:
                temp_net = [
                    nn.LeakyReLU(TaskConfig().enc_leaky_relu),
                    nn.Conv1d(
                        hidden_channel, hidden_channel,
                        kernel_size=kr[i], dilation=d,
                        padding=get_padding(kr[i], d))
                ]
                self.net.append(nn.Sequential(*temp_net))
        self.net = nn.ModuleList(self.net)

    def forward(self, x):
        for res_block in self.net:
            x_temp = res_block(x)
            x = x + x_temp
        return x


class MRF(nn.Module):

    def __init__(self, hidden_channel, kr, Dr):
        super(MRF, self).__init__()

        self.len_kr = len(kr)

        self.net = []
        for krs, Drs in zip(kr, Dr):
            self.net.append(ResBlock(hidden_channel, kr, Drs))
        self.net = nn.ModuleList(self.net)

    def forward(self, x):
        x_res = None
        for res_block in self.net:
            x_temp = res_block(x)
            if x_res is None:
                x_res = x_temp
            else:
                x_res = x_res + x_temp
        return x_res / self.len_kr
