import math
import os
import random
import torch
import torch.nn as nn
import torchaudio
import torch.utils.data

import librosa
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

from utils.config import TaskConfig

from utils.dataset.load_dataset import load_dataset

silence = -11.5129251


class MelSpec(nn.Module):
    def __init__(self, config=TaskConfig()):
        super(MelSpec, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sampling_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.fmin,
            f_max=config.fmax,
            pad=(config.n_fft - config.hop_length) // 2,
            n_mels=config.num_mels,
            center=False
        )
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(
            librosa.filters.mel(
                sr=config.sampling_rate,
                n_fft=config.n_fft,
                n_mels=config.num_mels,
                fmin=config.fmin,
                fmax=config.fmax
            ).T
        ))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spectrogram(audio).clamp_(min=1e-5).log_()
        return mel


def mel_len(batch):
    return batch.waveform_length / TaskConfig().hop_length
