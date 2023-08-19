import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.adv_utils import pgd_whitebox
from utils.general_utils import AverageMeter, split_data_and_move_to_device, get_accuracy


def get_output_for_batch(model, img, temp=1):
    """
        model(x) is expected to return logits (instead of softmax probas)
    """
    with torch.no_grad():
        out = nn.Softmax(dim=-1)(model(img) / temp)
        p, index = torch.max(out, dim=-1)
    return p.data.cpu().numpy(), index.data.cpu().numpy()


def base(model, device, val_loader):
    """
        Evaluating on unmodified validation set inputs.
    """
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")

    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = split_data_and_move_to_device(data, device)
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure get_accuracy and record loss
            acc1, acc5 = get_accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    return top1.avg


def adv(model, device, val_loader, args):
    """
        Evaluate on adversarial validation set inputs.
    """

    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
    adv_top5 = AverageMeter("Adv-Acc_5", ":6.2f")
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = split_data_and_move_to_device(data, device)
            result = model(images)
            output = result
            loss = criterion(output, target)

            acc1, acc5 = get_accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if args.epsilon_test != 0 and args.num_steps_test != 0:
                # adversarial images
                images = pgd_whitebox(
                    model,
                    images,
                    target,
                    device,
                    args.epsilon_test,
                    args.num_steps_test,
                    args.step_size_test,
                )

            # compute output
            result = model(images)

            output = result
            loss = criterion(output, target)

            # measure get_accuracy and record loss
            acc1, acc5 = get_accuracy(output, target, topk=(1, 5))
            adv_losses.update(loss.item(), images.size(0))
            adv_top1.update(acc1[0], images.size(0))
            adv_top5.update(acc5[0], images.size(0))

    return adv_top1.avg


def std_val(model, router, device, val_loader):
    """
        Evaluating on unmodified validation set inputs.
    """
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")

    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()
    router.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = split_data_and_move_to_device(data, device)
            output = model(images)
            loss = criterion(output, target)

            # measure get_accuracy and record loss
            acc1, acc5 = get_accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    return top1.avg


def adv_val_router(model, device, val_loader, args):
    """
        Evaluate on adversarial validation set inputs.
    """
    router_adv_top1 = AverageMeter("Acc_1", ":6.2f")
    router_adv_losses = AverageMeter("Adv_Loss", ":.4f")
    adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    model.eval()
    model.router.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = split_data_and_move_to_device(data, device)
            router_target = target % args.n_expert

            if args.epsilon_test != 0 and args.num_steps_test != 0:
                # adversarial images
                x_pgd = Variable(images.data, requires_grad=True)
                random_noise = (
                    torch.FloatTensor(x_pgd.shape).uniform_(-args.epsilon_test, args.epsilon_test).to(device)
                )
                x_pgd = Variable(x_pgd.data + random_noise, requires_grad=True)

                for _ in range(args.num_steps_test):
                    with torch.enable_grad():
                        scores_adv = model.router(x_pgd)
                        loss = torch.nn.CrossEntropyLoss()(scores_adv, router_target)
                    loss.backward()
                    eta = args.step_size_test * x_pgd.grad.data.sign()
                    x_pgd = Variable(x_pgd.data + eta, requires_grad=True)
                    eta = torch.clamp(x_pgd.data - images.data, -args.epsilon_test, args.epsilon_test)
                    x_pgd = Variable(images.data + eta, requires_grad=True)
                    x_pgd = Variable(torch.clamp(x_pgd, 0.0, 1.0), requires_grad=True)
            else:
                x_pgd = images

            # compute output
            adv_output = model(x_pgd)
            adv_scores = model.scores
            adv_loss = criterion(adv_output, target)
            adv_router_loss = criterion(adv_scores, router_target)

            # measure get_accuracy and record loss
            acc1 = torch.argmax(adv_output, 1).eq(target).float().mean().item() * 100
            adv_losses.update(adv_loss.item(), images.size(0))
            adv_top1.update(acc1, images.size(0))
            router_acc1 = torch.argmax(adv_scores, 1).eq(router_target).float().mean().item() * 100
            router_adv_losses.update(adv_router_loss.item(), images.size(0))
            router_adv_top1.update(router_acc1, images.size(0))

    return adv_top1.avg, router_adv_top1.avg


def adv_val(model, router, device, val_loader, args):
    """
        Evaluate on adversarial validation set inputs.
    """

    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
    adv_top5 = AverageMeter("Adv-Acc_5", ":6.2f")
    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    model.eval()
    router.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = split_data_and_move_to_device(data, device)

            if args.epsilon_test != 0 and args.num_steps_test != 0:
                # adversarial images
                x_pgd = Variable(images.data, requires_grad=True)
                random_noise = (
                    torch.FloatTensor(x_pgd.shape).uniform_(-args.epsilon_test, args.epsilon_test).to(device)
                )
                x_pgd = Variable(x_pgd.data + random_noise, requires_grad=True)

                for _ in range(args.num_steps_test):
                    with torch.enable_grad():
                        scores_adv = router(x_pgd)
                        out = model(x_pgd)
                        loss = torch.nn.CrossEntropyLoss()(out, target)
                        loss += args.beta * torch.nn.CrossEntropyLoss()(scores_adv, target % args.n_expert)
                    loss.backward()
                    eta = args.step_size_test * x_pgd.grad.data.sign()
                    x_pgd = Variable(x_pgd.data + eta, requires_grad=True)
                    eta = torch.clamp(x_pgd.data - images.data, -args.epsilon_test, args.epsilon_test)
                    x_pgd = Variable(images.data + eta, requires_grad=True)
                    x_pgd = Variable(torch.clamp(x_pgd, 0.0, 1.0), requires_grad=True)
            else:
                x_pgd = images

            # compute output
            result = model(x_pgd)

            output = result
            loss = criterion(output, target)

            # measure get_accuracy and record loss
            acc1, acc5 = get_accuracy(output, target, topk=(1, 5))
            adv_losses.update(loss.item(), images.size(0))
            adv_top1.update(acc1[0], images.size(0))
            adv_top5.update(acc5[0], images.size(0))

    return adv_top1.avg
