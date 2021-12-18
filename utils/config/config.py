import torch
import dataclasses
import os


@dataclasses.dataclass
class TaskConfig:
    work_dir: str = "./dla4_hfg"  # pix2pix2 directory
    work_dir_dataset: str = "./datasets/LJSpeech"  # dataset directory
    model_path: str = "./models"  # path to save future models
    results_location: str = "./results"
    dataset_full_name: str = os.path.join(work_dir_dataset, "LJSpeech-1.1")
    input_wavs_dir: str = os.path.join(dataset_full_name, "wavs")

    train_share: float = 0.7

    input_training_file: str = os.path.join(work_dir, "utils", "dataset", "overfit_batch.txt")
    input_validation_file: str = os.path.join(work_dir, "utils", "dataset", "overfit_batch.txt")

    dataset_url: str = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

    use_scheduler: bool = True
    no_val: bool = False  # set True if you don't want to run validation during training

    torch_seed: int = 42  # set torch seed for reproduction
    num_epochs: int = 2500
    batch_size: int = 8
    batch_limit: int = -1  # set number of batches that will be used in training

    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')  # set device

    wandb: bool = True  # set False if you don't want to send logs to wandb
    wandb_api: str = ""  # wandb api
    wandb_project: str = "dla4_hfg"  # wandb project name
    log_result_every_iteration: int = 5  # set -1 if you don't want to log any results
    log_loss_every_iteration: int = 5  # set -1 if you don't want to log any loss information
    log_audio: bool = True

    save_models_every_epoch: int = 3  # model will be saved every save_models_every_epoch'th epoch

    lr_decay: float = 0.9

    betas: tuple = (0.8, 0.99)  # Adam betas
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5

    eps: float = 1e-9

    # Dataset config

    segment_size: int = 8192
    n_fft: int = 1024
    num_mels: int = 80
    hop_size: int = 256
    win_size: int = 1024
    fmin: int = 0
    fmax: int = 8000
    fmax_loss: int = None
    input_mels_dir: str = 'ft_dataset'
    output_dir: str = 'result'
    sampling_rate: int = 22050

    # HiFiGun config

    def __init__(self):
        # Generator
        self.enc_in_channels = 80

        self.enc_leaky_relu = 0.1

        # self.hu = 128  # V2
        self.hu = 512  # V1
        self.ku = [16, 16, 4, 4]
        self.kr = [3, 7, 11]
        self.Dr = [[[1, 1], [3, 1], [5, 1]], [[1, 1], [3, 1], [5, 1]], [[1, 1], [3, 1], [5, 1]]]

        #Discriminator
        #MPD
        self.mpd_p = [2, 3, 5, 7, 11]
        self.mpd_hidden_k = 5
        self.mpd_s = [3, 3, 3, 3, 1]
        self.mpd_hidden_channels = [1, 32, 128, 512, 1024]
        self.mpd_relu = 0.1

        # MSD
        self.msd_hidden_channels = [1, 128, 128, 256, 512, 1024, 1024]
        self.msd_hidden_k = [15, 41, 41, 41, 41, 41, 5]
        self.msd_s = [1, 2, 2, 4, 4, 1, 1]
        self.msd_g = [1, 4, 16, 16, 16, 16, 1]
        self.msd_relu = 0.1