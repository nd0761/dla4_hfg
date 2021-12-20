import torch
import os

from utils.model.generator import Generator
from IPython import display

from utils.config import TaskConfig
from utils.dataset.mel_dataset import MelSpec

import librosa


def save_audio(wav, file_path):
    wav = display.Audio(wav.cpu(), rate=TaskConfig().sampling_rate)
    with open(file_path, "wb") as f:
        f.write(wav.data)


def main_worker():
    print("initialize generator")
    model_gen = Generator().to(TaskConfig().device)
    model_gen.load_state_dict(torch.load(
        os.path.join(TaskConfig().model_path, "best_model_generator"),
        map_location=TaskConfig().device))
    model_gen.eval()

    file_names = os.listdir(TaskConfig().work_dir_test_dataset)

    featurizer = MelSpec().to(TaskConfig().device)

    test_f = []
    for file_name in file_names:
        if file_name[-1] != "v":
            continue
        file_path = os.path.join(TaskConfig().work_dir_test_dataset, file_name)
        wav, _ = librosa.load(file_path, sr=TaskConfig().sampling_rate)
        wav = torch.Tensor(wav).unsqueeze(0).to(TaskConfig().device)
        test_f.append(featurizer(wav).to(TaskConfig().device))

    for t_mel, file_name in zip(test_f, file_names):
        save_audio(model_gen(t_mel), os.path.join(TaskConfig().results_location, file_name))


if __name__ == "__main__":
    main_worker()
