import os
import itertools

import torch
import torch.nn as nn
from utils.config import TaskConfig, TransformerConfig

from utils.trainer.train_discriminator import train
from utils.model.generator import Generator
from utils.model.discriminator import MSDModel, MPDModel
from utils.dataset.mel import get_dataloader

from utils.vcoder import Vocoder

from utils.dataset.mel_dataset import MelSpec


from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

if TaskConfig().wandb:
    from utils.logger.wandb_log_utils import initialize_wandb


def main_worker():
    print("set torch seed")
    config = TaskConfig()
    print(config.device)
    torch.manual_seed(config.torch_seed)

    print("initialize filelist")
    # training_filelist, validation_filelist = get_dataset_filelist()

    print("initialize dataset")

    # train_dataset = MelDataset(training_filelist)

    # val_dataset = MelDataset(validation_filelist)

    print("initialize dataloader")

    train_loader = get_dataloader()

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config.batch_size
    # )
    print("Train size:", len(train_loader) * TaskConfig().batch_size, len(train_loader))
    # print("Val size:", len(val_dataset), len(val_loader))

    print("initialize model")
    model_generator = Generator().to(config.device)
    model_generator.to(config.device)
    model_mpd = MPDModel().to(config.device)
    model_mpd.to(config.device)
    model_msd = MSDModel().to(config.device)
    model_msd.to(config.device)
    print("generator parameters:", sum(param.numel() for param in model_generator.parameters()))
    print("MPD parameters:", sum(param.numel() for param in model_mpd.parameters()))
    print("MSD parameters:", sum(param.numel() for param in model_msd.parameters()))

    print("initialize optimizer")
    opt_gen = AdamW(
        model_generator.parameters(),
        lr=config.learning_rate,
        betas=TaskConfig().betas, eps=TaskConfig().eps
    )
    opt_dis = AdamW(
        itertools.chain(model_mpd.parameters(), model_msd.parameters()),
        lr=config.learning_rate,
        betas=TaskConfig().betas, eps=TaskConfig().eps
    )

    print("initialize scheduler")
    scheduler_gen = ExponentialLR(opt_gen, gamma=config.lr_decay, last_epoch=-1)
    scheduler_dis = ExponentialLR(opt_dis, gamma=config.lr_decay, last_epoch=-1)
    # scheduler = None
    wandb_session = None
    if config.wandb:
        wandb_session = initialize_wandb(config)

    print("initialize featurizer")
    featurizer = MelSpec().to(TaskConfig().device)

    print("start train procedure")

    train(
        model_generator,
        model_mpd, model_msd,
        opt_gen, opt_dis,
        train_loader, None,
        featurizer,
        scheduler_gen=scheduler_gen,
        scheduler_dis=scheduler_dis,
        save_model=False,
        config=config, wandb_session=wandb_session
    )


if __name__ == "__main__":
    main_worker()
