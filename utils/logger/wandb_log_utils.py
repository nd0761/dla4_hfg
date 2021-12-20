import wandb
from utils.config import TaskConfig
from IPython import display

import matplotlib.pyplot as plt

import os


def initialize_wandb(config=TaskConfig()):
    wandb_session = wandb.init(project=config.wandb_project, entity="nd0761")
    wandb.config = config.__dict__
    return wandb_session


def log_audio(wandb_session, wav, tmp_path, wandb_result, delete_res=True):
    with open(tmp_path, "wb") as f:
        f.write(wav.data)
    wandb_session.log({wandb_result: wandb.Audio(tmp_path, sample_rate=TaskConfig().sampling_rate)})
    if delete_res:
        os.remove(tmp_path)


def log_melspec(wandb_session, melspec, wandb_result):
    plt.imshow(melspec.detach().squeeze(0))
    wandb_session.log({
        wandb_result: plt
    })


def log_wandb_audio(config, wandb_session, vocoder, melspec_predict,
                    melspec_gt, wv_gt,
                    log_type="train"):
    #     log_melspec(wandb_session, melspec_predict, log_type + ".predict_melspec")
    reconstructed_wav = vocoder(melspec_predict).detach().squeeze(1)
    wav = display.Audio(reconstructed_wav.cpu(), rate=TaskConfig().sampling_rate)
    tmp_path = config.work_dir + "temp.wav"
    log_audio(wandb_session, wav, tmp_path, log_type + ".audio_on_predict_mel")

    #     log_melspec(wandb_session, melspec_gt, log_type + ".gt_melspec")
    reconstructed_wav = vocoder(melspec_gt).detach().squeeze(1)
    wav = display.Audio(reconstructed_wav.cpu(), rate=TaskConfig().sampling_rate)
    tmp_path = config.work_dir + "temp.wav"
    log_audio(wandb_session, wav, tmp_path, log_type + ".audio_on_original_mel")

    gt_wav = display.Audio(wv_gt.cpu(), rate=TaskConfig().sampling_rate)
    log_audio(wandb_session, gt_wav, tmp_path, log_type + ".audio_original_wav")
