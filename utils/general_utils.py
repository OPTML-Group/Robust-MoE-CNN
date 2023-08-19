import math
import os
import random
import shutil
import sys
from distutils.dir_util import copy_tree

import numpy as np
import torch
import yaml

import data
import models
from data.huggingface import prepare_huggingface_data


def set_router(model, router):
    model.router = router
    for module in model.modules():
        if hasattr(module, 'router'):
            module.router = router


def set_seed(seed, deterministic=False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    torch.backends.cudnn.enabled = not deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def create_save_dir(args, result_dir, special_prefix=None):
    result_sub_dir_name = "{}-{}-ratio{}-epochs{}-lr{}-eps{}-steps{}-beta{}".format(
        args.dataset,
        args.arch,
        args.ratio,
        args.epochs,
        args.lr,
        int(args.epsilon * 255),
        args.num_steps,
        args.beta
    )
    if "moe" in args.arch:
        result_sub_dir_name = result_sub_dir_name + "-expert{}-".format(
            args.n_expert,
        )
    result_sub_dir_name = result_sub_dir_name + f"-seed{args.seed}"
    if special_prefix is not None:
        result_sub_dir_name = special_prefix + "_" + result_sub_dir_name
    result_sub_dir = os.path.join(
        result_dir,
        result_sub_dir_name
    )
    create_subdirs(result_sub_dir)
    print('Saving to %s' % result_sub_dir)
    return result_sub_dir


def save_checkpoint(
        state, is_best, result_dir, filename="checkpoint.pth.tar"
):
    create_subdirs(result_dir)
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, "model_best.pth.tar"),
        )


def create_subdirs(sub_dir):
    """
    sub_dir: the directory to be created
    force: if the True, the existing sub_dir (if existed) will be cleaned; otherwise an Error will be thrown
    """
    os.makedirs(sub_dir, exist_ok=True)
    # os.makedirs(os.path.join(sub_dir, "checkpoint"), exist_ok=True)


def write_to_file(file, data, option):
    with open(file, option) as f:
        f.write(data)


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)


# ref:https://github.com/allenai/hidden-networks/blob/master/configs/parser.py
def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


# ref: https://github.com/allenai/hidden-networks/blob/master/args.py
def parse_configs_file(args):
    # get commands from command line
    override_args = argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.configs).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.configs}")
    args.__dict__.update(loaded_yaml)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.val, global_step)


def get_accuracy(output, target, topk=(1,)):
    """Computes the get_accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def split_data_and_move_to_device(data, device):
    def process(x):
        return x[0].to(device) if isinstance(x, list) else x.to(device)

    if isinstance(data, dict):
        results = [process(x) for x in data.values()]
    elif isinstance(data, list):
        results = [process(x) for x in data]
    return results


def prepare_data(args):
    if args.dataset == "CIFAR10":
        dataset = data.CIFAR10(args)
        image_dim = 32
        train_loader, train_router_loader, test_loader, normalize, num_classes = dataset.data_loaders()
    elif args.dataset == "CIFAR10C":
        train_loader, train_router_loader, test_loader, normalize, num_classes = data.cifar10c_dataloaders(
            args.corrupt_type, args.data_dir)
        image_dim = 32
    elif args.dataset == "CIFAR100":
        dataset = data.CIFAR100(args)
        image_dim = 32
        train_loader, train_router_loader, test_loader, normalize, num_classes = dataset.data_loaders()
    elif args.dataset == "TinyImageNet":
        image_dim = 128
        loader, info = prepare_huggingface_data(dataset="tiny_imagenet", path=args.data_dir, resolution=image_dim,
                                                batch_size=args.batch_size, num_workers=8)
        train_loader, train_router_loader, test_loader = loader["train"], loader["train_router"], loader["val"]
        num_classes = info["num_classes"]
        normalize = info["normalize"]
    elif args.dataset == "ImageNet":
        image_dim = 224
        loader, info = prepare_huggingface_data(dataset="imagenet", path=args.data_dir, resolution=image_dim,
                                                batch_size=args.batch_size, num_workers=8)
        train_loader, train_router_loader, test_loader = loader["train"], loader["train_router"], loader["val"]
        num_classes = info["num_classes"]
        normalize = info["normalize"]
    else:
        raise NotImplementedError

    return train_loader, train_router_loader, test_loader, normalize, num_classes, image_dim


def get_data_model(args, device):
    train_loader, train_router_loader, test_loader, normalize, num_classes, image_dim = prepare_data(args)
    # ori models will ignore the parameter n_expert
    model = models.__dict__[args.arch](num_classes=num_classes, n_expert=args.n_expert, ratio=args.ratio).to(device)
    if args.normalize:
        model.normalize = normalize.to(device)

    return model, train_loader, train_router_loader, test_loader, image_dim


def initialize_weights(model="kaiming_normal", init_type="kaiming_normal"):
    assert init_type in ["kaiming_normal", "kaiming_uniform", "signed_const"]
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if init_type == "signed_const":
                n = math.sqrt(
                    2.0 / (m.kernel_size[0] * m.kernel_size[1] * m.in_channels)
                )
                m.weight.data = m.weight.data.sign() * n
            elif init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
            if init_type == "signed_const":
                n = math.sqrt(2.0 / m.in_features)
                m.weight.data = m.weight.data.sign() * n
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
