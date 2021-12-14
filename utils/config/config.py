import torch
import dataclasses


@dataclasses.dataclass
class TaskConfig:
    work_dir: str = "./dla4_hfg"  # pix2pix2 directory
    work_dir_dataset: str = "./datasets"  # dataset directory
    model_path: str = "./models"  # path to save future models
    results_location: str = "./results"

    use_scheduler: bool = True
    no_val: bool = False  # set True if you don't want to run validation during training

    torch_seed: int = 42  # set torch seed for reproduction
    num_epochs: int = 200
    batch_size: int = 1
    batch_limit: int = -1  # set number of batches that will be used in training

    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')  # set device

    wandb: bool = True  # set False if you don't want to send logs to wandb
    wandb_api: str = ""  # wandb api
    wandb_project: str = "dla4_hfg"  # wandb project name
    log_result_every_iteration: int = 5  # set -1 if you don't want to log any results
    log_loss_every_iteration: int = 5  # set -1 if you don't want to log any loss information

    save_models_every_epoch: int = 3  # model will be saved every save_models_every_epoch'th epoch

    betas: tuple = (0.5, 0.999)  # Adam betas
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5

    # HiFiGun config

    def __init__(self):
        self.enc_in_channels = 80
        self.enc_out_channels = 1
        self.dec_in_channels = 1
        self.dec_out_channels = 1

        self.enc_leaky_relu = 0.2
        self.enc_kernel = 3
        self.enc_dilation = (1, 3, 5)

        self.hu = 128  # V2
        self.ku = [16, 16, 4, 4]  # V2
        self.kr = [3, 7, 11]  # V2
        self.Dr = [[[1, 1], [3, 1], [5, 1]], [[1, 1], [3, 1], [5, 1]], [[1, 1], [3, 1], [5, 1]]]  # V2
