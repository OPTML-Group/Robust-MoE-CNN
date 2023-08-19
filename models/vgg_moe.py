import torch
import torch.nn as nn

from models.layers.moe_layer import MoEConv, MoEBase


class VGG(MoEBase):
    def __init__(self, cfgs, batch_norm, num_classes=10, n_expert=5, ratio=1.0):
        super(VGG, self).__init__()
        self.cfgs = cfgs
        self.features = self.make_layers(MoEConv, n_expert, ratio, batch_norm=batch_norm)
        last_conv_channels_len = [i for i in cfgs if isinstance(i, int)][-1]
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(int(last_conv_channels_len * ratio) * 2 * 2, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if self.router is not None:
            self.set_score(self.router(x))
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self, conv_layer, n_expert, ratio, batch_norm=True):
        layers = []
        in_channels = 3
        for depth, v in enumerate(self.cfgs):
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = int(ratio * v)
                if depth == 0:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                else:
                    conv2d = conv_layer(in_channels, v, kernel_size=3, padding=1, bias=False, n_expert=n_expert)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


cfgs = {
    "2": [64, "M", 64, "M"],
    "4": [64, 64, "M", 128, 128, "M"],
    "6": [64, 64, "M", 128, 128, "M", 256, 256, "M"],
    "8": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M"],
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
}


def vgg2_mix(**kwargs):
    return VGG(cfgs["2"], batch_norm=False, **kwargs)


def vgg2_bn_mix(**kwargs):
    return VGG(cfgs["2"], batch_norm=True, **kwargs)


def vgg4_mix(**kwargs):
    return VGG(cfgs["4"], batch_norm=False, **kwargs)


def vgg4_bn_mix(**kwargs):
    return VGG(cfgs["4"], batch_norm=True, **kwargs)


def vgg6_moe(**kwargs):
    return VGG(cfgs["6"], batch_norm=False, **kwargs)


def vgg6_bn_moe(**kwargs):
    return VGG(cfgs["6"], batch_norm=True, **kwargs)


def vgg8_moe(**kwargs):
    return VGG(cfgs["8"], batch_norm=False, **kwargs)


def vgg8_bn_moe(**kwargs):
    return VGG(cfgs["8"], batch_norm=True, **kwargs)


def vgg11_moe(**kwargs):
    return VGG(cfgs["11"], batch_norm=False, **kwargs)


def vgg11_bn_moe(**kwargs):
    return VGG(cfgs["11"], batch_norm=True, **kwargs)


def vgg13_moe(**kwargs):
    return VGG(cfgs["13"], batch_norm=False, **kwargs)


def vgg13_bn_moe(**kwargs):
    return VGG(cfgs["13"], batch_norm=True, **kwargs)


def vgg16_moe(**kwargs):
    return VGG(cfgs["16"], batch_norm=False, **kwargs)


def vgg16_bn_moe(**kwargs):
    return VGG(cfgs["16"], batch_norm=True, **kwargs)
