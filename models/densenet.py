#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 7:51 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
from jittor import nn, Module
import math
import jittor as jt

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm(in_planes)
        self.conv1 = nn.Conv(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm(4*growth_rate)
        self.conv2 = nn.Conv(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def execute(self, x):
        out = self.conv1(nn.relu(self.bn1(x)))
        out = self.conv2(nn.relu(self.bn2(out)))
        out =  jt.transpose(out, (1,0,2,3))
        x = jt.transpose(x, (1,0,2,3))
        out = jt.concat([out,x], 0)
        out = jt.transpose(out, (1, 0, 2, 3))
        #out = jt.reshape(out, [x.shape[0],-1,out.shape[2],out.shape[3]])
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm(in_planes)
        self.conv = nn.Conv(in_planes, out_planes, kernel_size=1, bias=False)
        self.pool = nn.Pool(2)
    def execute(self, x):
        out = self.conv(nn.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

        self.pool = nn.Pool(4)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def execute(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = nn.relu(self.bn(out))
        out = self.pool(out)
        out = out.reshape([out.shape[0], -1])
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

