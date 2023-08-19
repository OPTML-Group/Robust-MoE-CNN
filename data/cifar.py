import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from data.normalize import NormalizeByChannelMeanStd

num_workers = 4


class CIFAR10C(torch.utils.data.Dataset):
    def __init__(self, root, name_list, train, transform=None):

        idxs = np.random.permutation(50000 if train else 10000)
        batch_idxs = np.array_split(idxs, len(name_list))

        for idx, name in enumerate(name_list):
            data_path = os.path.join(root, 'train' if train else 'test', name + '.npy')
            target_path = os.path.join(root, 'train' if train else 'test', 'labels.npy')

            if hasattr(self, 'data'):
                self.data[batch_idxs[idx]] = np.load(data_path)[batch_idxs[idx]]
                self.targets[batch_idxs[idx]] = np.int64(np.load(target_path))[batch_idxs[idx]]
            else:
                self.data = np.load(data_path)
                self.targets = np.int64(np.load(target_path))

        self.transform = transform

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        return self.transform(img), targets

    def __len__(self):
        return len(self.data)


def cifar10c_dataloaders(name, root, val_ratio=0.0, batch_size=128, num_workers=2):
    if not isinstance(name, list):
        name = [name]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    val_num = int(val_ratio * 50000)
    train_num = 50000 - val_num
    idxs_train = sorted(random.sample(list(range(50000)), train_num))
    idxs_val = sorted(list(set(range(50000)) - set(idxs_train)))

    train_whole_set = CIFAR10C(root, name, train=True, transform=train_transform)
    train_set = Subset(train_whole_set, idxs_train)
    val_set = Subset(train_whole_set, idxs_val)
    test_set = CIFAR10C(root, name, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    router_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    normalization = NormalizeByChannelMeanStd(
        mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])

    num_classes = 10

    return train_loader, router_train_loader, test_loader, normalization, num_classes


class CIFAR10:
    """
        CIFAR-10 dataset.
    """

    def __init__(self, args):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        valset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        valset.data = torch.flip(torch.tensor(valset.data), dims=[0]).numpy()
        valset.targets = torch.flip(torch.tensor(valset.targets), dims=[0]).numpy()

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs
        )

        val_loader = DataLoader(
            valset,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs
        )

        testset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

        num_classes = 10

        return train_loader, val_loader, test_loader, dataset_normalization, num_classes


class CIFAR10cluster(torch.utils.data.Dataset):

    def __init__(self, args, type='train'):
        self.args = args

        self.tr_train = [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [
            transforms.ToPILImage(),
            transforms.ToTensor()]

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        if type == 'train':
            self.transform = self.tr_train
        else:
            self.transform = self.tr_test

        idx_file = torch.load(os.path.join(args.dataidxfile, 'pred_train' if type == 'train' else 'pred_test'),
                              map_location="cpu")

        data = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True if type == 'train' else False,
            download=True,
            transform=self.tr_train,
        )
        self.imgs = data.data
        self.targets = data.targets
        self.cluster = idx_file

    def __getitem__(self, index):
        return self.transform(self.imgs[index]), [self.targets[index], self.cluster[index]]

    def __len__(self):
        return len(self.imgs)


class CIFAR100:
    """
        CIFAR-100 dataset.
    """

    def __init__(self, args):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR100"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        trainset_twin = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR100"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        trainset_twin.data = torch.flip(torch.tensor(trainset_twin.data), dims=[0]).numpy()
        trainset_twin.targets = torch.flip(torch.tensor(trainset_twin.targets), dims=[0]).numpy()

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            **kwargs
        )

        train_loader_twin = DataLoader(
            trainset_twin,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            **kwargs
        )

        testset = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR100"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=num_workers, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )

        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

        num_classes = 100

        return train_loader, train_loader_twin, test_loader, dataset_normalization, num_classes
