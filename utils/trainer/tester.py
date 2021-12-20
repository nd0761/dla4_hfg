import torch
from utils.logger.wandb_log_utils import log_wandb_audio


@torch.no_grad()
def test(
        model_generator,
        tests,
        config,
        wandb_session=None
):
    model_generator.eval()
    res = []
    for t_mel in tests:
        pred = model_generator(t_mel.unsqueeze(0))
        if wandb_session is not None:
            log_wandb_audio(config, wandb_session, model_generator,
                            pred, t_mel.unsqueeze(0), None, log_type="test")
        else:
            res.append(pred)
    if not config.wandb:
        return res
