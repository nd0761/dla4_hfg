from utils.config import TaskConfig
from utils.loss import gen_loss, feat_loss

from torch.nn.modules.loss import MSELoss, L1Loss
import os

if TaskConfig().wandb:
    from utils.logger.wandb_log_utils import log_wandb_audio

from utils.dataset import mel_spectrogram
import torch

from utils.loss import dis_loss, feat_loss, gen_loss
from tqdm import tqdm


@torch.no_grad()
def test(
        model_generator,
        loader,
        config=TaskConfig(), wandb_session=None,
        mode="test"
):
    model_generator.eval()

    for i, batch in enumerate(loader):
        if config.batch_limit != -1 and i >= config.batch_limit:
            break

        mels, waveform, filename, mel_loss = batch
        waveform = waveform.to(config.device)
        mels = mels.to(config.device)

        predict = model_generator(mels)
        pred_mel = mel_spectrogram(predict.squeeze(1))

        if config.wandb and i % config.log_result_every_iteration == 0:
            model_generator.eval()
            for m_i in range(len(pred_mel)):
                log_wandb_audio(config, wandb_session, model_generator,
                                pred_mel[m_i].unsqueeze(0), mels[m_i].unsqueeze(0),
                                waveform[m_i], log_type="test")

