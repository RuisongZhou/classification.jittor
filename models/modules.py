#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 4:26 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import jittor as jt
from jittor import init, Module, nn
import numpy as np
import math


class DWConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DWConv, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups
        self.group_channel_in = in_channels // groups
        self.group_channel_out = out_channels // groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        Kh, Kw = self.kernel_size
        self.weight = init.relu_invariant_gauss([groups, out_channels//groups, in_channels // groups, Kh, Kw], dtype="float",
                                                mode="fan_out")
        if bias:
            self.bias = init.uniform([out_channels], dtype="float", low=-1, high=1)
        else:
            self.bias = None

    def execute(self, x):
        N, C, H, W = x.shape
        Kh, Kw = self.kernel_size
        oh = (H + self.padding[0] * 2 - Kh * self.dilation[0] + self.dilation[0] - 1) // self.stride[0] + 1
        ow = (W + self.padding[1] * 2 - Kw * self.dilation[1] + self.dilation[1] - 1) // self.stride[1] + 1

        x = jt.reshape(x,[N,self.groups, self.group_channel_in, H, W])
        xx = x.reindex(
            [N, self.groups, self.group_channel_out, self.group_channel_in, oh, ow, Kh, Kw], [
                'i0',  # Nid
                'i1', # Group
                'i3',  # Cid
                f'i4*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}',  # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}',  # Wid+KWid
            ])
        ww = self.weight.broadcast(xx.shape, [0, 4, 5])
        yy = xx * ww
        y = yy.sum([3, 6, 7])  # Kc, Kh, Kw
        y = jt.reshape(y,[N, self.out_channels, oh, ow])
        if self.bias is not None:
            b = self.bias.broadcast(y.shape, [0, 2, 3])
            y = y + b
        return y

if __name__ == '__main__':
    def test():
        img = np.random.rand(1,32,16,16)
        img = jt.array(img)
        conv = nn.Conv(32,64,3,1,padding=1,groups=1)
        res = conv(img)
        print(res.shape)
        print(type(res))

    test()