import os.path

from utils.config import TaskConfig
from utils.loss import gen_loss, feat_loss

from torch.nn.modules.loss import MSELoss, L1Loss
import os

if TaskConfig().wandb:
    from utils.logger.wandb_log_utils import log_wandb_audio

from utils.dataset import MelDataset, load_wav, dynamic_range_compression, dynamic_range_decompression, \
    dynamic_range_compression_torch, dynamic_range_decompression_torch, spectral_normalize_torch, \
    spectral_de_normalize_torch, mel_spectrogram, get_dataset_filelist
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_epoch(
        model_generator,
        opt_gen,
        loader, scheduler,
        loss_fn,
        config=TaskConfig(), wandb_session=None
):
    model_generator.train()
    for param in model_generator.parameters():
        param.requires_grad = True

    losses_gen = 0.
    len_batch = 0

    for i, batch in enumerate(loader):
        if config.batch_limit != -1 and i >= config.batch_limit:
            break
        len_batch += 1

        mels, waveform, _, _ = batch
        waveform = waveform.to(config.device)
        mels = mels.to(config.device)

        predict = model_generator(mels)
        pred_mel = mel_spectrogram(predict.squeeze(1))

        opt_gen.zero_grad()

        l_g = loss_fn(mels, pred_mel) * 45
        l_g.backward()
        opt_gen.step()

        losses_gen += l_g.detach().item()

        if scheduler is not None:
            scheduler.step()
        if config.wandb and i % config.log_loss_every_iteration == 0 and config.wandb:
            if scheduler is not None:
                a = scheduler.get_last_lr()[0]
                wandb_session.log({
                    "train.lr": scheduler.get_last_lr()[0]
                })
            wandb_session.log({
                "train.loss_gen": l_g.detach().cpu().numpy()
            })
        if config.wandb and i % config.log_result_every_iteration == 0:
            model_generator.eval()
            for m_i in range(len(mels)):
                log_wandb_audio(config, wandb_session, model_generator,
                                pred_mel[m_i].unsqueeze(0), mels[m_i].unsqueeze(0), waveform[m_i])
            model_generator.train()
    return losses_gen / len_batch


@torch.no_grad()
def validation(
        model_generator,
        loader,
        loss_fn,
        config=TaskConfig(), wandb_session=None,
        mode="val"
):
    model_generator.eval()

    val_losses_gen = 0.
    len_batch = 0

    for i, batch in enumerate(loader):
        if config.batch_limit != -1 and i >= config.batch_limit:
            break
        len_batch += 1

        mels, waveform, filename, mel_loss = batch
        waveform = waveform.to(config.device)
        mels = mels.to(config.device)

        predict = model_generator(mels)
        pred_mel = mel_spectrogram(predict.squeeze(1))

        l_g = loss_fn(mels, pred_mel) * 45
        val_losses_gen += l_g.item()

        if config.wandb and i % config.log_loss_every_iteration == 0 and config.wandb:
            wandb_session.log({
                "val.loss_gen": l_g.detach().cpu().numpy()
            })
        if config.wandb and i % config.log_result_every_iteration == 0:
            model_generator.eval()
            for m_i in range(len(mels)):
                log_wandb_audio(config, wandb_session, model_generator,
                                pred_mel[m_i].unsqueeze(0), mels[m_i].unsqueeze(0), waveform[m_i], log_type="val")

    return val_losses_gen


# TODO: update test_results_log for discriminator
# @torch.no_grad()
# def test_results_log(
#         model_generator,
#         loader,
#         image_location=TaskConfig().image_location,
#         config=TaskConfig(),
#         wandb_session=None,
#         mode="test", save_images=True,
#         save_as_three=TaskConfig().save_as_three
# ):
#     model_generator.eval()
#
#     mean = (-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5)
#     std = (1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5)
#     unnormalize = transforms.Normalize(mean, std)
#
#     # for param in model_generator.parameters():
#     #     param.requires_grad = False
#
#     # if config.use_tqdm:
#     #     for i, (input_img, ground_truth_img) in tqdm(enumerate(loader), desc="Test"):
#     ind = 0
#     for i, (input_img, ground_truth_img) in enumerate(loader):
#         input_img = input_img.to(config.device)
#         ground_truth_img = ground_truth_img.to(config.device)
#
#         predicted_img = model_generator(input_img)
#
#         # else save to local directory
#         directories = [os.path.join(image_location, "tricolor")]
#
#         for directory in directories:
#             if not os.path.exists(directory):
#                 os.makedirs(directory)
#             for input_img1, ground_truth_img1, predicted_img1 in zip(input_img, ground_truth_img, predicted_img):
#                 tricolor = torch.cat((unnormalize(predicted_img1), unnormalize(input_img1)), 2)
#                 if config.dataset_name == "restore":
#                     tricolor = torch.cat((unnormalize(input_img1), unnormalize(predicted_img1)), 2)
#                 if save_as_three:
#                     tricolor = torch.cat((unnormalize(input_img1),
#                                           unnormalize(ground_truth_img1),
#                                           unnormalize(predicted_img1)),
#                                          2)
#                 save_image(tricolor, os.path.join(directory, str(ind + 1) + ".png"))
#         ind += 1


def save_best_model(config, current_loss_gen, new_loss_gen, model_gen):
    if current_loss_gen < 0 or new_loss_gen < current_loss_gen:
        print("UPDATING BEST MODEL GENERATOR , NEW BEST LOSS:", new_loss_gen)
        best_model_path = os.path.join(config.model_path, "best_model_generator")
        torch.save(model_gen.state_dict(), best_model_path)
        return new_loss_gen
    return current_loss_gen


def train(
        model_generator,
        opt_gen,
        train_loader, val_loader,
        scheduler=None,
        save_model=False, model_path=None,
        config=TaskConfig(), wandb_session=None
):
    best_loss_gen = -1.

    gen_loss_fun = nn.L1Loss()

    # criterion = L1Loss()
    for n in range(config.num_epochs):
        gen_loss = train_epoch(
            model_generator,
            opt_gen,
            train_loader, scheduler,
            gen_loss_fun,
            config, wandb_session)

        print("GEN LOSS", gen_loss)

        best_loss_gen = save_best_model(config, best_loss_gen, gen_loss, model_generator)

        if n % config.save_models_every_epoch == 0:
            model_path = os.path.join(config.model_path, "model_epoch_gen")
            torch.save(model_generator.state_dict(), model_path)

        if not config.no_val:
            validation(
                model_generator,
                val_loader,
                gen_loss_fun,
                config, wandb_session
            )
        if config.wandb:
            wandb_session.log({"epoch": n})
        print('\n------\nEND OF EPOCH', n, "\n------\n")
    if save_model:
        torch.save(model_generator.state_dict(), os.path.join(config.model_path, "final_gen"))
