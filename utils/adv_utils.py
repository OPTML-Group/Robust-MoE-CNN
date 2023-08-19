import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def pgd_loss(model, x_natural, y, device, optimizer, step_size, epsilon, perturb_steps, ):
    ce_criterion = torch.nn.CrossEntropyLoss()

    if epsilon != 0 and perturb_steps != 0:
        x_adv = pgd_whitebox(model, x_natural, y, device, epsilon, perturb_steps, step_size, )

    else:
        x_adv = x_natural

    model.train()
    optimizer.zero_grad()
    loss_robust = ce_criterion(model(x_adv), y)

    return loss_robust


def trades_loss(model, x_natural, y, device, optimizer, step_size, epsilon, perturb_steps, clip_min=0.0, clip_max=1.0,
        distance="l_inf", natural_criterion=nn.CrossEntropyLoss(), ):
    if epsilon != 0 and perturb_steps != 0:

        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction="sum")
        model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = (x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach())
        if distance == "l_inf":
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    out_adv = model(x_adv)
                    out_nat = model(x_natural)

                    loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1), F.softmax(out_nat, dim=1), )
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, clip_min, clip_max)
        else:
            raise NotImplementedError

        model.train()

        x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
        optimizer.zero_grad()

        logits_nat = model(x_natural)
        loss_natural = natural_criterion(logits_nat, y)
        logits_adv = model(x_adv)

        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

    else:
        model.train()
        optimizer.zero_grad()
        logits_nat = model(x_natural)
        loss_natural = natural_criterion(logits_nat, y)
        loss_robust = torch.tensor(0)

    return loss_natural, loss_robust


# TODO: support L-2 attacks too.
def pgd_whitebox(model, x, y, device, epsilon, num_steps, step_size, clip_min=0.0, clip_max=1.0, is_random=True, ):
    x_pgd = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (torch.FloatTensor(x_pgd.shape).uniform_(-epsilon, epsilon).to(device))
        x_pgd = Variable(x_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            result = model(x_pgd)
            try:
                out, _ = result
            except:
                out = result
            loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        eta = step_size * x_pgd.grad.data.sign()
        x_pgd = Variable(x_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
        x_pgd = Variable(x.data + eta, requires_grad=True)
        x_pgd = Variable(torch.clamp(x_pgd, clip_min, clip_max), requires_grad=True)

    return x_pgd


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)
