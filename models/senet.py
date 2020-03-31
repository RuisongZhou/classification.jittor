#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 8:04 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
from jittor import nn, Module
import jittor as jt

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(planes)
            )

        # SE layers
        self.fc1 = nn.Conv(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv(planes//16, planes, kernel_size=1)

    def execute(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = nn.pool(out, out.shape[2],'maximum',0)
        w = nn.relu(self.fc1(w))
        w = nn.Sigmoid()(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = nn.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm(in_planes)
        self.conv1 = nn.Conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv(planes//16, planes, kernel_size=1)
        self.act = nn.Sigmoid()
    def execute(self, x):
        out = nn.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(nn.relu(self.bn2(out)))

        # Squeeze
        w = nn.pool(out, out.shape[2],'maximum',0)
        w = nn.relu(self.fc1(w))
        w = self.act(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.pool = nn.Pool(4)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def execute(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.reshape([out.shape[0], -1])
        out = self.linear(out)
        return out


def SENet18():
    return SENet(PreActBlock, [2,2,2,2])