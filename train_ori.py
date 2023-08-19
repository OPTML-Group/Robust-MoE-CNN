from __future__ import absolute_import
from __future__ import print_function

import logging
import os
import sys
import time

import torch
from tqdm import tqdm

from args import get_args_parser
from utils.adv_utils import trades_loss
from utils.eval_utils import adv as adv_val
from utils.eval_utils import base as std_val
from utils.general_utils import (
    save_checkpoint, AverageMeter, split_data_and_move_to_device,
    parse_configs_file, create_save_dir, initialize_weights,
    set_seed, get_data_model)
from utils.schedules import get_lr_policy, get_optimizer


def train_single_epoch(model, device, train_loader, epoch, args, optimizer):
    print(f" ->->->->->->->->->-> Epoch {epoch} with Adversarial training (TRADES) <-<-<-<-<-<-<-<-<-<-")

    losses = AverageMeter("Loss", ":.4f")
    losses_natural = AverageMeter("Loss-natural", ":.3f")
    losses_robust = AverageMeter("Loss-robust", ":.3f")
    top1 = AverageMeter("Acc_1", ":6.2f")

    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch} Training", ncols=120)
    model.train()
    for data in pbar:
        images, target = split_data_and_move_to_device(data, device)

        result = model(images)

        # calculate robust loss
        loss_natural, loss_robust = trades_loss(
            model=model,
            x_natural=images,
            y=target,
            device=device,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
        )

        loss = loss_natural + args.beta * loss_robust

        # measure get_accuracy and record loss
        with torch.no_grad():
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            losses_natural.update(loss_natural.item(), batch_size)
            losses_robust.update(loss_robust.item(), batch_size)
            top1.update(torch.argmax(result, 1).eq(target).float().mean().item(), batch_size)

        pbar.set_postfix_str(
            f"Source Acc {100 * top1.avg:.2f}%, Loss {losses_natural.avg:.5f}, Robust Loss{losses_robust.avg:.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    args = get_args_parser().parse_args()
    if args.configs is not None:
        parse_configs_file(args)

    # create result dir (for logs, checkpoints, etc.)
    if args.evaluate:
        result_sub_dir = os.path.join("results", "evaluate")
    else:
        result_sub_dir = os.path.join("results", "training")
    result_sub_dir = os.path.join(result_sub_dir, os.path.basename(__file__.split('.')[0]))
    result_sub_dir = create_save_dir(args, result_sub_dir, special_prefix=args.exp_identifier)

    # add logger
    set_seed(args.seed)

    # Set logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a"))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(args)

    # Select device
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare data and model
    model, train_loader, train_router_loader, test_loader, image_dim = get_data_model(args, device)
    initialize_weights(model)

    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args.lr, args.epochs)

    # Record the best get_accuracy
    start_epoch = 0
    best_acc = 0  # RA determines the best acc (epoch) if adv evaluation is used, otherwise sa
    sa_record = 0  # This records the SA of the best epoch.

    # resume (if checkpoint provided).
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            assert checkpoint["router_type"] == args.router_type
            start_epoch = checkpoint["epoch"]
            best_acc = checkpoint["best_acc"]
            sa_record = checkpoint["sa_record"]
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("=> resuming from '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            raise ValueError("=> No checkpoint found at '{}' for resume, please double check!".format(args.resume))

    # Evaluate
    if args.evaluate:
        sa = std_val(model, device, test_loader)
        ra = adv_val(model, device, test_loader, args)
        logger.info(f"Evaluation results: SA: {sa: .2f}%, RA: {ra: .2f}%.")
        return

    # Start training
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        lr_policy(epoch)

        # train
        train_single_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
        )

        sa = std_val(model, device, test_loader)
        ra = adv_val(model, device, test_loader, args)
        is_best = ra > best_acc
        if is_best:
            best_acc = ra
            sa_record = sa
        logger.info(
            f"Epoch {epoch}, SA: {sa: .2f}%, RA: {ra: .2f}%. [best performance (RA): {best_acc: .2f}, (SA): {sa_record: .2f}]"
        )

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "sa_record": sa_record,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            result_dir=os.path.join(result_sub_dir, "checkpoint"),
        )

        epoch_end_time = time.time()
        logger.info(f"Time consumption for current epoch is {(epoch_end_time - epoch_start_time):.2f}s")


if __name__ == "__main__":
    main()
