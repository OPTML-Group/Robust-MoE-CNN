import numpy as np
import torch


def get_lr_policy(lr_schedule):
    """Implement a new scheduler directly in this file.
    Args should contain a single choice for learning rate scheduler."""

    d = {
        "cosine": cosine_schedule,
        "step": step_schedule,
    }
    return d[lr_schedule]


def get_optimizer(model, args):
    if args.optimizer == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise NotImplementedError(f"{args.optimizer} is not supported.")
    return optim


def set_new_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def cosine_schedule(optimizer, lr, epochs):
    def set_lr(epoch, lr=lr, epochs=epochs):
        a = lr * 0.5 * (1 + np.cos((epoch - 1) / epochs * np.pi))

        set_new_lr(optimizer, a)

    return set_lr


def step_schedule(optimizer, lr, epochs):
    def set_lr(epoch, lr=lr, epochs=epochs):

        a = lr
        if epoch >= 0.75 * epochs:
            a = lr * 0.1
        if epoch >= 0.9 * epochs:
            a = lr * 0.01
        if epoch >= epochs:
            a = lr * 0.001

        set_new_lr(optimizer, a)

    return set_lr
