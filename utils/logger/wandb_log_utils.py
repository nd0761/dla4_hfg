import wandb
from utils.config import TaskConfig
from IPython import display

import os


def initialize_wandb(config=TaskConfig()):
    wandb_session = wandb.init(project=config.wandb_project, entity="nd0761")
    wandb.config = config.__dict__
    return wandb_session


def log_audio(wandb_session, wav, tmp_path, wandb_result, delete_res=True):
    with open(tmp_path, "wb") as f:
        f.write(wav.data)
    wandb_session.log({wandb_result: wandb.Audio(tmp_path, sample_rate=22050)})
    if delete_res:
        os.remove(tmp_path)


def log_wandb_audio(batch, config, wandb_session, vocoder, melspec_predict, log_type="train",
                    ground_truth=True):
    reconstructed_wav = vocoder.inference(melspec_predict).cpu()
    wav = display.Audio(reconstructed_wav, rate=22050)
    tmp_path = config.work_dir + "temp.wav"
    log_audio(wandb_session, wav, tmp_path, log_type + ".audio_predict")
    if ground_truth:
        gt_wav = display.Audio(batch.waveform[0].cpu(), rate=22050)
        log_audio(wandb_session, gt_wav, tmp_path, log_type + ".audio_original")
