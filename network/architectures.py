# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import math
import copy
from .base import *
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, pr=0.):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward2(self, x, residual):
        out = self.relu(x)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)


class SResNet(DCPBasicClass):
    # 32 layers ResNet for CIFAR-10

    def __init__(self, num_classes=10, pr=0.3):
        super(SResNet, self).__init__(pr=pr)
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.channels.append(0)
        self.layer1 = self._make_layer(BasicBlock, 16, 5, skip=False)
        self.layer2 = self._make_layer(BasicBlock, 32, 5, stride=2, skip=False)
        self.layer3 = self._make_layer(BasicBlock, 64, 5, stride=2, skip=False)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)
        self.channel_utility = torch.tensor(self.channel_utility)
        print(self.channels)
        self._parameter_init()

    def _make_para(self, planes=16, skip=False):
        temp = torch.zeros(planes)
        temp.fill_(1./planes)
        if skip is True:
            self.activated_channels.append(planes)
        else:
            self.activated_channels.append(round(planes * (1 - self.pruned_rate)))
        self.channel_utility.extend(temp)
        self.channels.append(self.channels[-1] + planes)
        self.mask.append(torch.zeros(1, planes, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, skip=False):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pr=self.pruned_rate))
        self._make_para(planes=planes, skip=skip)

        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, pr=self.pruned_rate))
            self._make_para(planes=planes)

        return nn.Sequential(*layers)

    def forward(self, x):
        count = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for l, layer in enumerate([self.layer1, self.layer2, self.layer3]):
            for _, (_, m) in enumerate(layer._modules.items()):
                # 第一层，保留残差
                residual = x
                x = m.conv1(x)
                x = m.bn1(x)
                if self.training is True:
                    x.register_hook(self.compute_rank)
                    self.activation_stack.push(x)
                    self.layer_index_stack.push(count)
                x = x * self.mask[count].expand_as(x)
                # 中间层
                x = m.forward2(x, residual)
                count += 1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SVgg(DCPBasicClass):
    def __init__(self, pr=0.0, num_classes=10):
        super(SVgg, self).__init__(pr=pr)
        self.feature = self._make_layers(cfg=cfg['D'])
        self.classifier = nn.Linear(512, num_classes)
        self.channel_utility = torch.tensor(self.channel_utility)
        print(self.channels)
        self._parameter_init()

    def _make_layers(self, cfg=[], bn=True):
        layers = []
        in_channels = 3
        count = 0
        channel_count = 0
        self.channels.append(channel_count)
        for i, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                count += 1
            else:
                self.channel_utility.extend(torch.zeros(v))
                channel_count += v
                self.channels.append(channel_count)
                self.mask.append(torch.zeros(1, v, 1, 1))
                self.activated_channels.append(round(v * (1 - self.pruned_rate)))
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        count = 0
        for _, (_, m) in enumerate(self.feature._modules.items()):
            x = m(x)
            # cache the activations to update channel utility values
            if isinstance(m, nn.BatchNorm2d):
                if self.training is True:
                    x.register_hook(self.compute_rank)
                    self.activation_stack.push(x)
                    self.layer_index_stack.push(count)
                x = x * self.mask[count].expand_as(x)
                count += 1
        x = x.view(x.size(0), -1)
        return self.classifier(x)




