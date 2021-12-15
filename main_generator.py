import os

import torch
import torch.nn as nn
from utils.config import TaskConfig, TransformerConfig

from utils.trainer import train
from utils.model.generator import Generator
from utils.dataset import FacadesDataset

from torch.utils.data import DataLoader
from torch.optim import AdamW

if TaskConfig().wandb:
    from utils.logger.wandb_log_utils import initialize_wandb


def main_worker():
    print("set torch seed")
    config = TaskConfig()
    print(config.device)
    torch.manual_seed(config.torch_seed)

    print("initialize dataset")
    train_dataset = FacadesDataset(
        os.path.join(config.work_dir_dataset, config.dataset_name),
        None, None,
        split="train", flip=True, flip_prob=TransformerConfig().flip_prob)
    val_dataset = FacadesDataset(
        os.path.join(config.work_dir_dataset, config.dataset_name),
        None, None,
        split="val"
    )

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
        betas=(0.9, 0.98), eps=1e-9
    )

    print("initialize scheduler")
    scheduler = None
    wandb_session = None
    if config.wandb:
        wandb_session = initialize_wandb(config)

    print("start train procedure")

    train_baseline(
        model_generator,
        opt_gen,
        train_loader, val_loader,
        scheduler=scheduler,
        save_model=False,
        config=config, wandb_session=wandb_session
    )


if __name__ == "__main__":
    main_worker()
