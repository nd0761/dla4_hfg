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

from utils.loss import dis_loss, feat_loss, gen_loss
from tqdm import tqdm


def train_epoch(
        model_generator,
        model_mpd, model_msd,
        opt_gen, opt_dis,
        loader, scheduler,
        loss_fn,
        config=TaskConfig(), wandb_session=None
):
    model_generator.train()
    for param in model_generator.parameters():
        param.requires_grad = True

    losses_gen = 0.
    losses_dis = 0.
    len_batch = 0

    for i, batch in enumerate(loader):
        if config.batch_limit != -1 and i >= config.batch_limit:
            break
        len_batch += 1

        mels, waveform, _, _ = batch
        waveform = waveform.to(config.device).unsqueeze(1)
        mels = mels.to(config.device)

        predict = model_generator(mels)
        pred_mel = mel_spectrogram(predict.squeeze(1))

        for param in model_generator.parameters():
            param.requires_grad = False

        for param in model_msd.parameters():
            param.requires_grad = True
        for param in model_mpd.parameters():
            param.requires_grad = True

        opt_dis.zero_grad()
        mpd_real_res, mpd_gen_res, _, _ = model_mpd(waveform, predict.detach())
        mpd_sum_loss, _, _ = dis_loss(mpd_gen_res, mpd_real_res)

        msd_real_res, msd_gen_res, _, _ = model_msd(waveform, predict.detach())
        msd_sum_loss, _, _ = dis_loss(msd_gen_res, msd_real_res)

        l_d = mpd_sum_loss + msd_sum_loss
        l_d.backward()
        opt_dis.step()

        for param in model_generator.parameters():
            param.requires_grad = True

        for param in model_msd.parameters():
            param.requires_grad = False
        for param in model_mpd.parameters():
            param.requires_grad = False

        opt_gen.zero_grad()

        _, mpd_gen_res, mpd_real_features, mpd_gen_features = model_mpd(waveform, predict)
        _, msd_gen_res, msd_real_features, msd_gen_features = model_msd(waveform, predict)

        d_feat_loss = feat_loss(mpd_gen_features, mpd_real_features) + feat_loss(msd_gen_features, msd_real_features)

        d_gen_loss = gen_loss(mpd_gen_res)[0] + gen_loss(msd_gen_res)[0]

        lg = loss_fn(mels, pred_mel)
        #         print("\nLOSS\n", d_gen_loss.item(), d_feat_loss, lg)

        l_g = d_gen_loss + \
              TaskConfig().feat_loss_coef * d_feat_loss + \
              loss_fn(mels, pred_mel) * TaskConfig().gen_loss_coef
        l_g.backward()
        opt_gen.step()

        losses_gen += l_g.detach().item()
        losses_dis += l_d.detach().item()

        if scheduler is not None:
            scheduler.step()
        if config.wandb and i % config.log_loss_every_iteration == 0 and config.wandb:
            if scheduler is not None:
                a = scheduler.get_last_lr()[0]
                wandb_session.log({
                    "train.lr": scheduler.get_last_lr()[0]
                })
            wandb_session.log({
                "train.loss_gen": l_g.detach().cpu().numpy(),
                "train.loss_dis": l_d.detach().cpu().numpy()
            })
        if config.wandb and i % config.log_result_every_iteration == 0:
            model_generator.eval()
            m_i = 0
            log_wandb_audio(config, wandb_session, model_generator,
                            pred_mel[m_i].unsqueeze(0), mels[m_i].unsqueeze(0), waveform[m_i])
            model_generator.train()
    return losses_gen / len_batch, losses_dis / len_batch


@torch.no_grad()
def validation(
        model_generator,
        model_mpd, model_msd,
        loader,
        loss_fn,
        config=TaskConfig(), wandb_session=None,
        mode="val"
):
    model_generator.eval()
    model_mpd.eval()
    model_msd.eval()

    val_losses_gen = 0.
    val_losses_dis = 0.
    len_batch = 0

    for i, batch in enumerate(loader):
        if config.batch_limit != -1 and i >= config.batch_limit:
            break
        len_batch += 1

        mels, waveform, filename, mel_loss = batch
        waveform = waveform.to(config.device).unsqueeze(1)
        mels = mels.to(config.device)

        predict = model_generator(mels)
        pred_mel = mel_spectrogram(predict.squeeze(1))

        mpd_real_res, mpd_gen_res, mpd_real_features, mpd_gen_features = model_mpd(waveform, predict)
        mpd_sum_loss, _, _ = dis_loss(mpd_gen_res, mpd_real_res)

        msd_real_res, msd_gen_res, msd_real_features, msd_gen_features = model_msd(waveform, predict.detach())
        msd_sum_loss, _, _ = dis_loss(msd_gen_res, msd_real_res)

        l_d = mpd_sum_loss + msd_sum_loss

        d_feat_loss = feat_loss(mpd_gen_features, mpd_real_features) + feat_loss(msd_gen_features, msd_real_features)
        d_gen_loss = gen_loss(mpd_gen_res)[0] + gen_loss(msd_gen_res)[0]

        lg = loss_fn(mels, pred_mel)

        l_g = d_gen_loss + \
              TaskConfig().feat_loss_coef * d_feat_loss + \
              lg * TaskConfig().gen_loss_coef

        val_losses_gen += l_g.item()
        val_losses_dis += l_d.item()

        if config.wandb and i % config.log_loss_every_iteration == 0 and config.wandb:
            wandb_session.log({
                "val.loss_gen": l_g.detach().cpu().numpy(),
                "val.loss_dis": l_d.detach().cpu().numpy()
            })
        if config.wandb and i % config.log_result_every_iteration == 0:
            model_generator.eval()
            m_i = 0
            log_wandb_audio(config, wandb_session, model_generator,
                            pred_mel[m_i].unsqueeze(0), mels[m_i].unsqueeze(0), waveform[m_i], log_type="val")

    return val_losses_gen


def save_best_model(config, current_loss_gen, current_loss_dis, new_loss_gen, new_loss_dis, model_gen, model_mpd,
                    model_msd):
    if current_loss_gen < 0 or new_loss_gen < current_loss_gen:
        print("UPDATING BEST MODEL GENERATOR , NEW BEST LOSS:", new_loss_gen)
        print("UPDATING MODEL DISCRIMINATOR , NEW LOSS:", new_loss_gen)
        best_model_path = os.path.join(config.model_path, "best_model_generator")
        torch.save(model_gen.state_dict(), best_model_path)
        best_model_path = os.path.join(config.model_path, "best_model_mpd")
        torch.save(model_mpd.state_dict(), best_model_path)
        best_model_path = os.path.join(config.model_path, "best_model_msd")
        torch.save(model_msd.state_dict(), best_model_path)
        return new_loss_gen, new_loss_dis
    return current_loss_gen, current_loss_dis


def train(
        model_generator,
        model_mpd, model_msd,
        opt_gen, opt_dis,
        train_loader, val_loader,
        scheduler=None,
        save_model=False, model_path=None,
        config=TaskConfig(), wandb_session=None
):
    best_loss_gen = -1.
    best_loss_dis = -1.

    gen_loss_fun = nn.L1Loss()

    for n in tqdm(range(config.num_epochs), desc="TRAINING PROCESS", total=config.num_epochs):
        gen_loss, dis_loss = train_epoch(
            model_generator,
            model_mpd, model_msd,
            opt_gen, opt_dis,
            train_loader, scheduler,
            gen_loss_fun,
            config, wandb_session)

        print("GEN LOSS", gen_loss)
        print("DIS LOSS", dis_loss)

        best_loss_gen, best_loss_dis = save_best_model(
            config,
            best_loss_gen, best_loss_dis,
            gen_loss, dis_loss,
            model_generator,
            model_mpd, model_msd
        )

        if n % config.save_models_every_epoch == 0:
            model_path = os.path.join(config.model_path, "model_gen_epoch_gen")
            torch.save(model_generator.state_dict(), model_path)
            model_path = os.path.join(config.model_path, "model_mpd_epoch_gen")
            torch.save(model_mpd.state_dict(), model_path)
            model_path = os.path.join(config.model_path, "model_msd_epoch_gen")
            torch.save(model_msd.state_dict(), model_path)

        if not config.no_val:
            validation(
                model_generator,
                model_mpd, model_msd,
                val_loader,
                gen_loss_fun,
                config, wandb_session
            )
        if config.wandb:
            wandb_session.log({"epoch": n})
        print('\n------\nEND OF EPOCH', n, "\n------\n")
    if save_model:
        torch.save(model_generator.state_dict(), os.path.join(config.model_path, "final_gen"))
        torch.save(model_mpd.state_dict(), os.path.join(config.model_path, "final_mpd"))
        torch.save(model_msd.state_dict(), os.path.join(config.model_path, "final_msd"))
