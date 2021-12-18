import os
import random
from utils.config import TaskConfig


def split_train_val_files():
    with open(TaskConfig().metadata_path, encoding='utf-8') as file:
        lines = file.readlines()

        random.shuffle(lines)

        train_len = len(lines) * TaskConfig().train_share
        train_data = lines[:train_len]
        val_data = lines[train_len:]

        with open(TaskConfig().input_training_file, "w") as f:
            f.write(str(train_data))
        with open(TaskConfig().input_validation_file, "w") as f:
            f.write(str(val_data))
