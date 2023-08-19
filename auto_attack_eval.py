import argparse
import os

import torch
from tqdm import tqdm

from models.layers.router import build_router
from torchattacks import AutoAttack
from utils.general_utils import (
    initialize_weights,
    set_seed, get_data_model, )


def main():
    parser = argparse.ArgumentParser(description="AutoAttack Evaluation")
    # Data
    parser.add_argument("--dataset", type=str, choices=("CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet"),
                        default="CIFAR10")
    parser.add_argument("--data-dir", type=str, default="../data/", help="path to datasets")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--normalize", action="store_true", default=False, help="Data normalization?")
    parser.add_argument("--data-fraction", type=float, default=1.0, help="Fraction of images used from training set", )
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--arch", type=str, default="resnet18_cifar_mix", help="Model architecture")
    parser.add_argument("--source-net", type=str, help="Checkpoint which will be pruned/fine-tuned", default=None)
    parser.add_argument("--n-expert", default=None, type=int)
    parser.add_argument("--epsilon", default=8.0, type=float, help="perturbation")
    args = parser.parse_args()

    args.epsilon /= 255

    # add logger
    set_seed(args.seed)

    # Select device
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare data and model
    model, train_loader, train_router_loader, test_loader, image_dim = get_data_model(args, device)
    initialize_weights(model)
    router = build_router(num_experts=args.n_expert).to(device)
    model.router = router

    # source_net serves as the pretrained model for either router training or finetuning.
    if args.source_net:
        if os.path.isfile(args.source_net):
            print("=> loading checkpoint '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            router.load_state_dict(checkpoint["router"])
            model.load_state_dict(checkpoint["state_dict"])
            print("=> resuming from '{}' (epoch {})".format(args.source_net, checkpoint["epoch"]))
        else:
            raise ValueError("=> No checkpoint found at '{}' for source_net, please double check!".format(args.source_net))

    model.eval()

    correct = 0
    total = 0
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"Standard Evaluation", ncols=120)
    for ii, (images, labels) in tqdm(enumerate(pbar)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()
        pbar.set_postfix_str(f"Accuracy: {100 * correct / total:.2f}%")
    print('Natural accuracy: %.2f %%' % (100. * (correct / total).cpu().item()))

    attacker = AutoAttack(model, norm='Linf', eps=args.epsilon)
    attack_total = 0
    attack_correct = 0
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"AutoAttack Evaluation", ncols=120)
    for ii, (data, label) in tqdm(enumerate(pbar)):
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        label = label.to(device)
        if device != 'cpu':
            perturbed_data = attacker(data, label).cuda(device=device)
        else:
            perturbed_data = attacker(data, label)

        score = model(perturbed_data)
        _, predicted = torch.max(score, 1)
        attack_total += label.cpu().size(0)
        attack_correct += (predicted == label).sum()
        pbar.set_postfix_str(f"Robust Accuracy: {100 * attack_correct / attack_total:.2f}%")
    print(f'The robust accuracy against epsilon {args.epsilon} is {attack_correct / attack_total * 100}')


if __name__ == "__main__":
    main()
