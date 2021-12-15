import torch
import torch.nn as nn
from utils.config import TaskConfig
from utils.model.generator.blocks import MRF, get_padding


# source shw5
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.in_net = nn.Sequential(
            nn.Conv1d(TaskConfig().enc_in_channels, TaskConfig().hu, kernel_size=7, stride=1, padding=get_padding(7, 1))
        )

        self.net = []
        cur_channels = TaskConfig().hu
        next_channels = TaskConfig().hu
        for i in range(len(TaskConfig().ku)):
            kus = TaskConfig().ku[i]
            cur_channels = TaskConfig().hu // pow(2, i)
            next_channels = TaskConfig().hu // pow(2, i + 1)
            temp = [
                nn.LeakyReLU(TaskConfig().enc_leaky_relu),
                nn.ConvTranspose1d(
                    in_channels=cur_channels,
                    out_channels=next_channels,
                    stride=kus // 2,
                    kernel_size=kus,
                    padding=kus // 4  # ToDo
                ),
                MRF(next_channels, TaskConfig().kr, TaskConfig().Dr)
            ]
            self.net.append(nn.Sequential(*temp))
        self.net = nn.Sequential(*self.net)
        self.out = nn.Sequential(
            *[
                nn.LeakyReLU(TaskConfig().enc_leaky_relu),
                nn.Conv1d(next_channels, 1, kernel_size=7, stride=1, padding=get_padding(7, 1)),
                nn.Tanh()
            ]
        )

    def forward(self, x):
        x = self.in_net(x)
        x = self.net(x)
        x = self.out(x)
        return x
