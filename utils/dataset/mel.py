from utils.config import TaskConfig
from utils.dataset.dataset import LJSpeechCollator, LJSpeechDataset
from torch.utils.data import DataLoader


def get_dataloader(dataset=LJSpeechDataset, path=TaskConfig().work_dir_dataset,
                   batch_size=TaskConfig().batch_size, collate_fn=LJSpeechCollator,
                   limit=TaskConfig().batch_limit):
    ds = dataset(path)
    if limit != -1:
        ds = list(ds)[:limit * batch_size]
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn())
