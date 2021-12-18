import torch
import torch.nn as nn
from utils.config import TaskConfig
from utils.model.discriminator.blocks import MPDBlock, MSDBlock


class MPDModel(nn.Module):
    def __init__(self):
        super(MPDModel, self).__init__()

        self.net = []
        for p in TaskConfig().mpd_p:
            self.net.append(MPDBlock(p))
        self.net = nn.ModuleList(self.net)

    def forward(self, x_real, x_gen):
        real_res = []
        gen_res = []

        real_features = []
        gen_features = []

        #         print("START", x_real.shape, x_gen.shape)

        for layer in self.net:
            #             print("------", x_real.shape, x_gen.shape)
            x_real_res, real_feat = layer(x_real)
            x_gen_res, gen_feat = layer(x_gen)
            #             print(x_real.shape, x_gen.shape)

            real_res.append(x_real_res)
            gen_res.append(x_gen_res)

            real_features.append(real_feat)
            gen_features.append(gen_feat)
        return real_res, gen_res, real_features, gen_features


class MSDModel(nn.Module):
    def __init__(self):
        super(MSDModel, self).__init__()

        self.net = []
        for sp in [True, False, False]:
            temp = []
            if not sp:
                temp.append(nn.AvgPool1d(4, 2, padding=2))
            temp.append(MSDBlock(sp))
            self.net.append(nn.Sequential(*temp))
        self.net = nn.ModuleList(self.net)

    def forward(self, x_real, x_gen):
        real_res = []
        gen_res = []

        real_features = []
        gen_features = []

        for layer in self.net:
            x_real_res, real_feat = layer(x_real)
            x_gen_res, gen_feat = layer(x_gen)

            real_res.append(x_real_res)
            gen_res.append(x_gen_res)

            real_features.append(real_feat)
            gen_features.append(gen_feat)
        return real_res, gen_res, real_features, gen_features
