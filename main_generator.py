import os

import torch
import torch.nn as nn
from utils.config import TaskConfig, TransformerConfig

from utils.trainer import train
from utils.model.generator import Generator
from utils.dataset import MelDataset, load_wav, dynamic_range_compression, dynamic_range_decompression, \
    dynamic_range_compression_torch, dynamic_range_decompression_torch, spectral_normalize_torch, \
    spectral_de_normalize_torch, mel_spectrogram, get_dataset_filelist

from utils.vcoder import Vocoder


from torch.utils.data import DataLoader
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
    training_filelist, validation_filelist = get_dataset_filelist()

    print("initialize dataset")

    train_dataset = MelDataset(training_filelist)

    val_dataset = MelDataset(validation_filelist)

    print("initialize dataloader")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size
    )
    print("Train size:", len(train_dataset), len(train_loader))
    print("Val size:", len(val_dataset), len(val_loader))

    print("initialize model")
    model_generator = nn.DataParallel(Generator()).to(config.device)
    model_generator.to(config.device)
    print("model parameters:", sum(param.numel() for param in model_generator.parameters()))

    print("initialize optimizer")
    opt_gen = AdamW(
        model_generator.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=TaskConfig().betas, eps=TaskConfig().eps
    )

    print("initialize scheduler")
    scheduler = ExponentialLR(opt_gen, gamma=config.lr_decay, last_epoch=config.num_epochs)
    wandb_session = None
    if config.wandb:
        wandb_session = initialize_wandb(config)

    vocoder = None
    if config.log_audio:
        print("initialize vocoder")
        vocoder = Vocoder().to(config.device).eval()

    print("start train procedure")

    train(
        model_generator,
        opt_gen,
        train_loader, val_loader,
        scheduler=scheduler,
        save_model=False,
        config=config, wandb_session=wandb_session,
        vocoder=vocoder
    )


if __name__ == "__main__":
    main_worker()
