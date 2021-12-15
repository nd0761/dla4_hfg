import os
import wget
import tarfile
from utils.config import TaskConfig

# !wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# !tar -xjf LJSpeech-1.1.tar.bz2


def load():
    dataset_path = TaskConfig().work_dir_dataset
    file_path = os.path.join(TaskConfig().work_dir_dataset, TaskConfig().dataset_name)

    filename = "LJSpeech-1.1.tar.bz2"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if os.path.exists(file_path):
        return  # dataset already exists
    if not os.path.isfile(os.path.join(".", filename)):  # if archive not exists
        filename = wget.download(TaskConfig().dataset_url)

    temp = tarfile.open(filename)
    temp.extractall(file_path)
    temp.close()
