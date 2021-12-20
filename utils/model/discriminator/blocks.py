import torch.nn as nn
import torch
import torch.nn.functional as F

from utils.config import TaskConfig
from utils.model.generator.blocks import get_padding


class MPDBlock(nn.Module):

    def __init__(self, p=1):
        super(MPDBlock, self).__init__()

        self.p = p
        self.net = []

        cur_channel = TaskConfig().mpd_hidden_channels[0]

        for next_channel, stride in zip(TaskConfig().mpd_hidden_channels[1:] + [1024], TaskConfig().mpd_s):
            self.net.append(nn.Sequential(
                *[
                    nn.Conv2d(
                        cur_channel, next_channel,
                        kernel_size=TaskConfig().mpd_hidden_k,
                        stride=stride,
                        padding=get_padding(TaskConfig().mpd_hidden_k[0], 1)
                    ),
                    nn.LeakyReLU(TaskConfig().mpd_relu)
                ]
            ))
            cur_channel = next_channel
        self.net = nn.ModuleList(self.net)

        self.out = nn.Sequential(
            nn.Conv2d(cur_channel, 1, kernel_size=(3, 1))
        )

    def forward(self, x):
        batch_size, seq_len, temp = x.shape

        if temp % self.p != 0:
            x = F.pad(x, (0, self.p - (temp % self.p)), "reflect")
            temp += self.p - (temp % self.p)

        x = x.view(batch_size, seq_len, temp // self.p, self.p)

        features = []

        for layer in self.net:
            x = layer(x)
            features.append(x)
        x = self.out(x)

        features.append(x)

        return torch.flatten(x, 1, -1), features


from torch.nn.utils import weight_norm, spectral_norm


class MSDBlock(nn.Module):

    def __init__(self, use_sp=False):
        super(MSDBlock, self).__init__()

        self.use_sp = use_sp
        self.net = []

        if use_sp:
            norm = spectral_norm
        else:
            norm = weight_norm

        cur_channel = TaskConfig().msd_hidden_channels[0]

        for next_channel, stride, kernel, group in zip(
                TaskConfig().msd_hidden_channels[1:] + [1024],
                TaskConfig().msd_s,
                TaskConfig().msd_hidden_k,
                TaskConfig().msd_g):
            self.net.append(nn.Sequential(
                *[
                    nn.Conv1d(
                        cur_channel, next_channel,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,
                        groups=group
                    ),
                    nn.LeakyReLU(TaskConfig().msd_relu)
                ]
            ))
            cur_channel = next_channel
        self.net = nn.ModuleList(self.net)

        self.out = nn.Sequential(
            nn.Conv1d(cur_channel, 1, kernel_size=3, padding=3)
        )

    def forward(self, x):
        features = []

        for layer in self.net:
            x = layer(x)
            features.append(x)
        x = self.out(x)
        features.append(x)

        return torch.flatten(x, 1, -1), features
