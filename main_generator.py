import torch
import torch.nn as nn
from utils.config import TaskConfig

from utils.trainer.train import train
from utils.model.generator import Generator
from utils.dataset.mel import get_dataloader

from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from utils.dataset.mel_dataset import MelSpec

if TaskConfig().wandb:
    from utils.logger.wandb_log_utils import initialize_wandb


def main_worker():
    print("set torch seed")
    config = TaskConfig()
    print(config.device)
    torch.manual_seed(config.torch_seed)

    print("initialize dataloader")

    train_loader = get_dataloader()

    print("Train size:", len(train_loader) * TaskConfig().batch_size, len(train_loader))

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
    scheduler = None
    wandb_session = None
    if config.wandb:
        wandb_session = initialize_wandb(config)

    print("initialize featurizer")
    featurizer = MelSpec().to(TaskConfig().device)

    print("start train procedure")

    train(
        model_generator,
        opt_gen,
        train_loader, None,
        featurizer=featurizer,
        scheduler=scheduler,
        save_model=False,
        config=config, wandb_session=wandb_session
    )


if __name__ == "__main__":
    main_worker()
