import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import math
import copy
from .base import *

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class StructuredBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(StructuredBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def add_residual(self, x, y):
        y = self.conv3(y)
        y = self.bn3(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return self.relu(y)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        out = self.add_residual(identity, x)

        return out

class DcpResNet(DCPBasicClass):
    def __init__(self, pr):
        super(DcpResNet, self).__init__(pr=pr)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.channels.append(0)
        block = StructuredBottleneck
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)
        self.channel_utility = torch.tensor(self.channel_utility)
        self.activated_channels = torch.tensor(self.activated_channels)

        print(self.channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print(m.weight.data.shape)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, StructuredBottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_para(self, planes=64):
        temp = torch.zeros(planes)
        temp.fill_(1./planes)
        self.activated_channels.append(round(planes * (1 - self.pruned_rate)))
        self.channel_utility.extend(temp)
        self.channels.append(self.channels[-1] + planes)
        self.mask.append(torch.zeros(1, planes, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self._make_para(planes=planes)
        self._make_para(planes=planes)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            self._make_para(planes=planes)
            self._make_para(planes=planes)

        return nn.Sequential(*layers)

    def forward(self, x):
        count = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for m in self.modules():
            if isinstance(m, StructuredBottleneck):
                identity = x
                x = m.conv1(x)
                x = m.bn1(x)
                if self.training is True:
                    x.register_hook(self.compute_rank)
                    self.activation_stack.push(x)
                    self.layer_index_stack.push(count)
                x = x * self.mask[count].expand_as(x)
                x = m.relu(x)
                count += 1
                x = m.conv2(x)
                x = m.bn2(x)
                if self.training is True:
                    x.register_hook(self.compute_rank)
                    self.activation_stack.push(x)
                    self.layer_index_stack.push(count)
                x = x * self.mask[count].expand_as(x)
                x = m.relu(x)
                count += 1
                x = m.add_residual(identity, x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PrunedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, activated_planes=None, stride=1, downsample=None):
        super(PrunedBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, activated_planes[0])
        self.bn1 = nn.BatchNorm2d(activated_planes[0])
        self.conv2 = conv3x3(activated_planes[0], activated_planes[1], stride)
        self.bn2 = nn.BatchNorm2d(activated_planes[1])
        self.conv3 = conv1x1(activated_planes[1], planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def add_residual(self, x, y):
        y = self.conv3(y)
        y = self.bn3(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return self.relu(y)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        out = self.add_residual(identity, x)

        return out

class PrunedResNet(nn.Module):
    def __init__(self, net):
        super(PrunedResNet, self).__init__()
        self.activated_channels = net.activated_channels.tolist()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_count = 0
        block = PrunedBottleneck
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)
        self._parameter_init(net, len(self.activated_channels))
        self.proportion = 0.5

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        local_activated = self.activated_channels[self.layer_count:self.layer_count+2]
        layers.append(block(self.inplanes, planes, local_activated, stride=stride, downsample=downsample))
        self.layer_count += 2
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            local_activated = self.activated_channels[self.layer_count:self.layer_count+2]
            layers.append(block(self.inplanes, planes, local_activated))
            self.layer_count += 2

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _parameter_init(self, net, layers):
        count = 0
        indices = []
        for layer in range(layers):
            start = net.channels[layer]
            end = net.channels[layer+1]
            index = torch.argsort(net.channel_utility[start:end], descending=True)[:net.activated_channels[layer]]
            indices.append(index)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1.weight.data = net.bn1.weight.data
        self.bn1.bias.data = net.bn1.bias.data
        for m1, m2 in zip(self.modules(), net.modules()):
            if isinstance(m1, PrunedBottleneck):
                index = indices[count]
                # prune 
                print("---copying block index {}...".format(count+1))
                m1.conv1.weight.data = m2.conv1.weight.data[index, :, :, :]
                m1.bn1.weight.data = m2.bn1.weight.data[index]
                m1.bn1.bias.data = m2.bn1.bias.data[index]
                count += 1

                index = indices[count]
                print("---copying block index {}...".format(count + 1))
                temp = m2.conv2.weight.data[:, indices[count-1], :, :]
                m1.conv2.weight.data = temp[index, :, :, :]
                m1.bn2.weight.data = m2.bn2.weight.data[index]
                m1.bn2.weight.data = m2.bn2.bias.data[index]
                count += 1

                m1.conv3.weight.data = m2.conv3.weight.data[:, index, :, :]
                m1.bn3.weight.data = m2.bn3.weight.data
                m1.bn3.weight.data = m2.bn3.bias.data
                if m1.downsample is not None:
                    for dm1, dm2 in zip(m1.downsample.modules(), m2.downsample.modules()):
                        if isinstance(dm1, nn.Conv2d):
                            dm1.weight.data = dm2.weight.data
                        if isinstance(dm1, nn.BatchNorm2d):
                            dm1.weight.data = dm2.weight.data
                            dm1.bias.data = dm2.bias.data
                print("--------------------------")
            if isinstance(m1, nn.Linear):
                m1.weight.data = m2.weight.data
                m1.bias.data = m2.bias.data

        for m1 in self.modules():
            if isinstance(m1, nn.Conv2d):
                print(m1.weight.shape)

def pruning(net):
    return PrunedResNet(net=net)