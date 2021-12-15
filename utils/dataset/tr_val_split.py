import os
import random
from utils.config import TaskConfig

def split_train_val_files():
    file_names = os.listdir(TaskConfig().dataset_full_name)

    random.shuffle(file_names)
    train_len = len(file_names) * TaskConfig().train_share

    train_data = file_names[:train_len]
    val_data = file_names[train_len:]
    with open(TaskConfig().input_training_file, "w") as f:
        f.write(str(train_data))
    with open(TaskConfig().input_validation_file, "w") as f:
        f.write(str(val_data))
