import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

from utils.config import TaskConfig
from utils.dataset.dataset import LJSpeechCollator, LJSpeechDataset
from torch.utils.data import DataLoader

from utils.dataset.load_dataset import load_dataset


# source -  https://github.com/jik876/hifi-gan/blob/master/meldataset.py
def get_dataset_filelist():
    input_wavs_dir = os.path.join(TaskConfig().dataset_full_name, "wavs")
    with open(TaskConfig().input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(TaskConfig().input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


def get_dataloader(dataset=LJSpeechDataset, path=TaskConfig().work_dir_dataset,
                   batch_size=TaskConfig().batch_size, collate_fn=LJSpeechCollator,
                   limit=TaskConfig().batch_limit):
    ds = dataset(path)
    if limit != -1:
        ds = list(ds)[:limit * batch_size]
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn())
