import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, conv_layer, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
                (not self.equalInOut)
                and conv_layer(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
                or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, conv_layer, stride, dropRate=0.0, **kwargs):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            conv_layer, block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(
            self, conv_layer, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    conv_layer,
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
            self,
            depth=34,
            num_classes=10,
            widen_factor=10,
            dropRate=0.0,
            ratio=1.0
    ):
        super(WideResNet, self).__init__()
        conv_layer = nn.Conv2d
        linear_layer = nn.Linear
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.ratio = ratio
        self.normalize = None
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, int(nChannels[0] * self.ratio), kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(
            n, int(nChannels[0] * self.ratio), int(nChannels[1] * self.ratio), block, conv_layer, 1, dropRate
        )
        # 2nd block
        self.block2 = NetworkBlock(
            n, int(nChannels[1] * self.ratio), int(nChannels[2] * self.ratio), block, conv_layer, 2, dropRate
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, int(nChannels[2] * self.ratio), int(nChannels[3] * self.ratio), block, conv_layer, 2, dropRate
        )
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(int(nChannels[3] * self.ratio))
        self.relu = nn.ReLU(inplace=True)
        self.fc = linear_layer(int(nChannels[3] * self.ratio), num_classes)
        self.nChannels = int(nChannels[3] * self.ratio)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, linear_layer):
                m.bias.data.zero_()

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def wrn_28_10_cifar_ori(**kwargs):
    return WideResNet(depth=28, widen_factor=10, **kwargs)


def wrn_28_4_cifar_ori(**kwargs):
    return WideResNet(depth=28, widen_factor=4, **kwargs)


def wrn_34_10_cifar_ori(**kwargs):
    return WideResNet(depth=34, widen_factor=10, **kwargs)


if __name__ == "__main__":
    model = wrn_34_10_cifar_ori(num_classes=10)
    inputs = torch.randn(32, 3, 32, 32)
    outputs = model(inputs)
    print(outputs.shape)
