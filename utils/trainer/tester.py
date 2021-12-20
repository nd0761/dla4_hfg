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
    for i, (gt_wav, t_mel) in enumerate(tests):
        pred = model_generator(t_mel)
        if wandb_session is not None:
            log_wandb_audio(config, wandb_session, model_generator,
                            t_mel, t_mel, gt_wav, log_type="test" + str(i))
        else:
            res.append(pred)
    if not config.wandb:
        return res
