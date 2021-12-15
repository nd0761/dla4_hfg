import os
import wget
import tarfile
from utils.config import TaskConfig

# !wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# !tar -xjf LJSpeech-1.1.tar.bz2


def load_dataset():
    dataset_path = TaskConfig().work_dir_dataset

    filename = "LJSpeech-1.1.tar.bz2"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if os.path.exists(TaskConfig().dataset_full_name):
        return  # dataset already exists
    if not os.path.isfile(os.path.join(".", filename)):  # if archive not exists
        filename = wget.download(TaskConfig().dataset_url)

    temp = tarfile.open(filename)
    temp.extractall(dataset_path)
    temp.close()

