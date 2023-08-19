import os

import torch
import torchvision
from datasets.load import load_dataset
from torch.utils.data import DataLoader

from data.normalize import NormalizeByChannelMeanStd


def prepare_huggingface_data(dataset, path, resolution, batch_size=512, num_workers=8):
    path = os.path.join(path, "huggingface")
    if dataset == "imagenet":
        train_set = load_dataset("imagenet-1k", use_auth_token=True, split="train", cache_dir=path)
        validation_set = load_dataset("imagenet-1k", use_auth_token=True, split="validation", cache_dir=path)
        bigger_resolution = int(resolution * 256 / 224)
        normalize = NormalizeByChannelMeanStd(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))

        def train_transform(examples):
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
                torchvision.transforms.Resize((bigger_resolution, bigger_resolution)),
                torchvision.transforms.RandomCrop((resolution, resolution)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

        def validation_transform(examples):
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
                torchvision.transforms.Resize((bigger_resolution, bigger_resolution)),
                torchvision.transforms.CenterCrop((resolution, resolution)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

    elif dataset == "tiny_imagenet":
        train_set = load_dataset("Maysee/tiny-imagenet", use_auth_token=True, split="train", cache_dir=path)
        validation_set = load_dataset("Maysee/tiny-imagenet", use_auth_token=True, split="valid", cache_dir=path)
        normalize = NormalizeByChannelMeanStd(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        def train_transform(examples):
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
                torchvision.transforms.Resize((resolution, resolution)),
                torchvision.transforms.RandomCrop(resolution, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

        def validation_transform(examples):
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
                torchvision.transforms.Resize((resolution, resolution)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples
    else:
        raise NotImplementedError

    train_set.set_transform(transform=train_transform)
    validation_set.set_transform(transform=validation_transform)
    loaders = {
        'train': DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        'train_router': DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        'val': DataLoader(validation_set, batch_size=batch_size, num_workers=num_workers, shuffle=False),
    }
    info = {
        'num_classes': len(train_set.info.features['label'].names),
        'class_names': train_set.info.features['label'].names,
        'trainset_size': train_set.num_rows,
        'valset_size': validation_set.num_rows,
        'normalize': normalize
    }
    return loaders, info
