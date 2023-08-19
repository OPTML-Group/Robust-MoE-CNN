import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, cfgs, batch_norm, num_classes=10, ratio=1.0, **kwargs):
        super(VGG, self).__init__()
        self.features = make_layers(cfgs, nn.Conv2d, batch_norm=batch_norm, ratio=ratio)
        last_conv_channels_len = [i for i in cfgs if isinstance(i, int)][-1]
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(nn.Linear(int(last_conv_channels_len * ratio) * 2 * 2, 256), nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, num_classes), )
        self.ratio = ratio

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, conv_layer, batch_norm=True, ratio=1.0):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(ratio * v)
            conv2d = conv_layer(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {"2": [64, "M", 64, "M"], "4": [64, 64, "M", 128, 128, "M"], "6": [64, 64, "M", 128, 128, "M", 256, 256, "M"],
    "8": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M"],
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, ], }


def vgg2_ori(**kwargs):
    return VGG(cfgs["2"], batch_norm=False, **kwargs)


def vgg2_bn_ori(**kwargs):
    return VGG(cfgs["2"], batch_norm=True, **kwargs)


def vgg4_ori(**kwargs):
    return VGG(cfgs["4"], batch_norm=False, **kwargs)


def vgg4_bn_ori(**kwargs):
    return VGG(cfgs["4"], batch_norm=True, **kwargs)


def vgg6_ori(**kwargs):
    return VGG(cfgs["6"], batch_norm=False, **kwargs)


def vgg6_bn_ori(**kwargs):
    return VGG(cfgs["6"], batch_norm=True, **kwargs)


def vgg8_ori(**kwargs):
    return VGG(cfgs["8"], batch_norm=False, **kwargs)


def vgg8_bn_ori(**kwargs):
    return VGG(cfgs["8"], batch_norm=True, **kwargs)


def vgg11_ori(**kwargs):
    return VGG(cfgs["11"], batch_norm=False, **kwargs)


def vgg11_bn_ori(**kwargs):
    return VGG(cfgs["11"], batch_norm=True, **kwargs)


def vgg13_ori(**kwargs):
    return VGG(cfgs["13"], batch_norm=False, **kwargs)


def vgg13_bn_ori(**kwargs):
    return VGG(cfgs["13"], batch_norm=True, **kwargs)


def vgg16_ori(**kwargs):
    return VGG(cfgs["16"], batch_norm=False, **kwargs)


def vgg16_bn_ori(**kwargs):
    return VGG(cfgs["16"], batch_norm=True, **kwargs)


if __name__ == "__main__":
    model = vgg16_bn_ori(ratio=0.2)
    print(model)
    print(model(torch.randn(3, 3, 224, 224)).shape)
