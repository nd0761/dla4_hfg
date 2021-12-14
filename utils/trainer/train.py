import os.path

from utils.config import TaskConfig
from torch.nn.modules.loss import MSELoss, L1Loss
import os

if TaskConfig().wandb:
    from utils.logger.wandb_log_utils import log_wandb_pic, log_wandb_one_img
if TaskConfig().use_tqdm:
    from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_epoch(
        model_generator, model_discriminator,
        opt_gen, opt_dis,
        loader, scheduler,
        loss_gan, loss_l1,
        config=TaskConfig(), wandb_session=None
):
    model_generator.train()
    model_discriminator.train()

    losses_gen = 0.
    losses_gen_gan = 0.
    losses_dis = 0.
    len_batch = 0
    # if config.use_tqdm:
    #     for i, (input_img, ground_truth_img) in tqdm(enumerate(loader), desc="Train epoch"):
    for i, (input_img, ground_truth_img) in enumerate(loader):
        if config.batch_limit != -1 and i >= config.batch_limit:
            break
        len_batch += 1
        # model_generator.train()
        # model_discriminator.eval()
        for param in model_generator.parameters():
            param.requires_grad = True
        for param in model_discriminator.parameters():
            param.requires_grad = False

        input_img = input_img.to(config.device)
        ground_truth_img = ground_truth_img.to(config.device)

        opt_gen.zero_grad()

        predicted_img = model_generator(input_img)

        input_pred = torch.cat((input_img, predicted_img), 1).to(config.device)
        pred_dis = model_discriminator(input_pred)

        l_g = loss_gan(pred_dis, torch.ones(pred_dis.shape).to(config.device))
        l_l1 = loss_l1(predicted_img, ground_truth_img)
        loss_gen = l_g + config.loss_coef * l_l1

        loss_gen.backward()

        opt_gen.step()

        # model_generator.eval()
        # model_discriminator.train()
        for param in model_generator.parameters():
            param.requires_grad = False
        for param in model_discriminator.parameters():
            param.requires_grad = True

        opt_dis.zero_grad()

        input_pred = torch.cat((input_img, predicted_img.detach()), 1).to(config.device)
        pred_dis = model_discriminator(input_pred)
        input_gt = torch.cat((input_img, ground_truth_img), 1).to(config.device)
        gt_dis = model_discriminator(input_gt)

        loss_dis = \
            (
                loss_gan(pred_dis, torch.zeros(pred_dis.shape).to(config.device)) +
                loss_gan(gt_dis, torch.ones(gt_dis.shape).to(config.device))
            ) * 0.5

        loss_dis.backward()
        opt_dis.step()

        # model_generator.eval()
        # model_discriminator.eval()
        # for param in model_generator.parameters():
        #     param.requires_grad = False
        # for param in model_discriminator.parameters():
        #     param.requires_grad = False

        losses_dis += (loss_dis.item())
        losses_gen_gan += (l_g.item())
        losses_gen += (loss_gen.item())

        if scheduler is not None:
            scheduler.step()

        if config.wandb and i % config.log_loss_every_iteration == 0 and config.wandb:
            if scheduler is not None:
                a = scheduler.get_last_lr()[0]
                wandb_session.log({
                    "learning_rate": a
                })
            wandb_session.log({
                "train.loss_gen": loss_gen.detach().cpu().numpy(),
                "train.loss_dis": loss_dis.detach().cpu().numpy(),
                "train.loss_gen_gan": l_g.detach().cpu().numpy(),
                "train.loss_gen_l1": l_l1.detach().cpu().numpy()
            })
        if config.wandb and i % config.log_img_every_iteration == 0:
            tricolor = torch.cat((input_img[0], ground_truth_img[0], predicted_img[0]), 2)
            log_wandb_one_img(wandb_session, tricolor)
            # log_wandb_pic(wandb_session, input_img[0], predicted_img[0], ground_truth_img=ground_truth_img[0])
    return losses_gen / len_batch, losses_gen_gan / len_batch, losses_dis / len_batch


@torch.no_grad()
def validation(
        model_generator, model_discriminator,
        loader,
        loss_gan, loss_l1,
        config=TaskConfig(), wandb_session=None,
        mode="val", save_images=False
):
    model_generator.eval()
    model_discriminator.eval()
    # for param in model_generator.parameters():
    #     param.requires_grad = False
    # for param in model_discriminator.parameters():
    #     param.requires_grad = False

    val_losses_gen = []
    val_losses_dis = []
    for i, (input_img, ground_truth_img) in enumerate(loader):
        if config.batch_limit != -1 and i >= config.batch_limit:
            break
        input_img = input_img.to(config.device)
        ground_truth_img = ground_truth_img.to(config.device)

        predicted_img = model_generator(input_img)

        input_pred = torch.cat((input_img, predicted_img), 1).to(config.device)
        pred_dis = model_discriminator(input_pred)
        input_gt = torch.cat((input_img, ground_truth_img), 1).to(config.device)
        gt_dis = model_discriminator(input_gt)

        l_g = loss_gan(pred_dis, torch.ones(pred_dis.shape).to(config.device))
        l_l1 = loss_l1(predicted_img, ground_truth_img)
        loss_gen = l_g + config.loss_coef * l_l1

        loss_dis = \
            (
                loss_gan(pred_dis, torch.zeros(pred_dis.shape).to(config.device)) +
                loss_gan(gt_dis, torch.ones(gt_dis.shape).to(config.device))
            ) * 0.5

        val_losses_gen.append(loss_gen.detach().cpu().numpy())
        val_losses_dis.append(loss_dis.detach().cpu().numpy())

        if config.wandb and config.log_loss_every_iteration != -1 and i % config.log_loss_every_iteration == 0:
            wandb_session.log({
                "val.loss_gen": loss_gen.detach().cpu().numpy(),
                "val.loss_dis": loss_dis.detach().cpu().numpy(),
                "val.loss_gen_gan": l_g.detach().cpu().numpy(),
                "val.loss_gen_l1": l_l1.detach().cpu().numpy()
            })

        if config.dataset_name != "restore" and (config.save_img_every_iteration == -1 or i % config.save_img_every_iteration != 0 or not config.wandb):
            continue

        # log validation images only to wandb
        tricolor = torch.cat((input_img[0], ground_truth_img[0], predicted_img[0]), 2)
        log_wandb_one_img(wandb_session, tricolor, mode=mode)
        # log_wandb_pic(wandb_session, input_img[0], predicted_img[0], mode=mode, ground_truth_img=ground_truth_img[0])
    return val_losses_gen, val_losses_dis


# TODO: update test_results_log for discriminator
@torch.no_grad()
def test_results_log(
        model_generator,
        loader,
        image_location=TaskConfig().image_location,
        config=TaskConfig(),
        wandb_session=None,
        mode="test", save_images=True,
        save_as_three=TaskConfig().save_as_three
):
    model_generator.eval()

    mean = (-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5)
    std = (1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5)
    unnormalize = transforms.Normalize(mean, std)

    # for param in model_generator.parameters():
    #     param.requires_grad = False

    # if config.use_tqdm:
    #     for i, (input_img, ground_truth_img) in tqdm(enumerate(loader), desc="Test"):
    ind = 0
    for i, (input_img, ground_truth_img) in enumerate(loader):
        input_img = input_img.to(config.device)
        ground_truth_img = ground_truth_img.to(config.device)

        predicted_img = model_generator(input_img)

        # else save to local directory
        directories = [os.path.join(image_location, "tricolor")]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
            for input_img1, ground_truth_img1, predicted_img1 in zip(input_img, ground_truth_img, predicted_img):
                tricolor = torch.cat((unnormalize(predicted_img1), unnormalize(input_img1)), 2)
                if config.dataset_name == "restore":
                    tricolor = torch.cat((unnormalize(input_img1), unnormalize(predicted_img1)), 2)
                if save_as_three:
                    tricolor = torch.cat((unnormalize(input_img1),
                                          unnormalize(ground_truth_img1),
                                          unnormalize(predicted_img1)),
                                         2)
                save_image(tricolor, os.path.join(directory, str(ind + 1) + ".png"))
        ind += 1


def save_best_model(config, current_loss_gen, new_loss_gen, new_loss_dis, model_gen, model_dis):
    if current_loss_gen < 0 or new_loss_gen < current_loss_gen:
        print("UPDATING BEST MODEL GENERATOR , NEW BEST LOSS:", new_loss_gen)
        print("UPDATING MODEL DISCRIMINATOR , NEW LOSS:", new_loss_dis)
        best_model_path = os.path.join(config.model_path, "best_model_generator")
        torch.save(model_gen.state_dict(), best_model_path)
        best_model_path = os.path.join(config.model_path, "best_model_discriminator")
        torch.save(model_dis.state_dict(), best_model_path)
        return new_loss_gen
    return current_loss_gen


def train(
        model_generator, model_discriminator,
        opt_gen, opt_dis,
        train_loader, val_loader,
        scheduler=None,
        save_model=False, model_path=None,
        config=TaskConfig(), wandb_session=None
):
    best_loss_gen = -1.
    best_loss_dis = -1.

    # loss_gan = MSELoss()
    loss_gan = nn.BCEWithLogitsLoss()
    loss_l1 = L1Loss()

    if config.l1 == "l2":
        loss_l1 = MSELoss()

    # criterion = L1Loss()
    for n in range(config.num_epochs):
        gen_loss, gen_gan_loss, dis_loss = train_epoch(
            model_generator, model_discriminator,
            opt_gen, opt_dis,
            train_loader, scheduler,
            loss_gan, loss_l1,
            config, wandb_session)
        # gen_loss = sum(losses_gen) / len(losses_gen)
        # dis_loss = sum(losses_dis) / len(losses_dis)
        # gen_gan_loss = sum(losses_gen_gan) / len(losses_gen_gan)
        print("GEN LOSS", gen_loss)
        print("GEN GAN LOSS", gen_gan_loss)
        print("DIS LOSS", dis_loss)

        # if gen_gan_loss < 0.2 or dis_loss < 0.2:
        #     print("ERROR NEEDS TO BE RESOLVED")
        #     break

        best_loss_gen = save_best_model(config, best_loss_gen, gen_loss, dis_loss, model_generator, model_discriminator)

        if n % config.save_models_every_epoch == 0:
            model_path = os.path.join(config.model_path, "model_epoch_gen")
            torch.save(model_generator.state_dict(), model_path)
            model_path = os.path.join(config.model_path, "model_epoch_dis")
            torch.save(model_discriminator.state_dict(), model_path)

        if not config.no_val:
            validation(
                model_generator, model_discriminator,
                val_loader,
                loss_gan, loss_l1,
                config, wandb_session
            )
        if config.wandb:
            wandb_session.log({"epoch": n})
        print('\n------\nEND OF EPOCH', n, "\n------\n")
    if save_model:
        torch.save(model_generator.state_dict(), os.path.join(config.model_path, "final_gen"))
        torch.save(model_discriminator.state_dict(), os.path.join(config.model_path, "final_dis"))
