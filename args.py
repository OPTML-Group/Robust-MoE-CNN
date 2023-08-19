import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument("--configs", type=str, default=None, help="configs file", )
    parser.add_argument("--exp-identifier", default=None, type=str, help="special identifier to differentiate each trial", )
    parser.add_argument("--seed", type=int, default=1234, help="random seed")

    # Model
    parser.add_argument("--arch", type=str, help="Model architecture")
    parser.add_argument("--resume", type=str, default="", help="path to latest checkpoint (default:None)")

    # Data
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=("CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet"))
    parser.add_argument("--data-dir", type=str, default="../data/", help="path to datasets")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--normalize", action="store_true", default=False, help="Data normalization?")
    parser.add_argument("--data-fraction", type=float, default=0.9, help="Fraction of images used from training set", )

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=("sgd", "adam"))
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--wd", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=("step", "cosine"))

    parser.add_argument("--router-optimizer", type=str, default="sgd", choices=("sgd", "adam"))
    parser.add_argument("--router-lr", type=float, default=0.1, help="router learning rate")
    parser.add_argument("--router-lr-schedule", type=str, default="cosine", choices=("cosine", "step"))

    # Adversarial Generation
    parser.add_argument("--epsilon", default=8.0 / 255, type=float, help="perturbation")
    parser.add_argument("--num-steps", default=2, type=int, help="perturb number of steps")
    parser.add_argument("--step-size", default=5.0 / 255, type=float, help="perturb step size")
    parser.add_argument("--epsilon-test", default=8.0 / 255, type=float, help="perturbation")
    parser.add_argument("--num-steps-test", default=10, type=int, help="perturb number of steps")
    parser.add_argument("--step-size-test", default=2.0 / 255, type=float, help="perturb step size")
    parser.add_argument("--beta", default=6.0, type=float, help="regularization, i.e., 1/lambda in TRADES")
    parser.add_argument("--alpha", default=6.0, type=float)

    # Evaluate
    parser.add_argument("--evaluate", action="store_true", default=False, help="Evaluate model")

    # MoE
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--n-expert", default=None, type=int)

    return parser
