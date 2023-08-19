import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GetPruneSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        score_L1_norm = torch.norm(scores.flatten(start_dim=1, end_dim=-1), p=1, dim=1)
        _, idx = score_L1_norm.sort()
        j = int((1 - k) * scores.shape[0])

        # flat_out and out access the same memory.
        out = scores.clone()
        flat_out = out.flatten(start_dim=1, end_dim=-1)  # share the same memory
        flat_out[idx[:j], :] = 0
        flat_out[idx[j:], :] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SubnetConv(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0

    def set_remain_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, x):
        adj = GetPruneSubnet.apply(self.popup_scores.abs(), self.ratio)
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
