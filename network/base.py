# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import math
import copy
import time

# The Stack to cache the internal outputs
class Stack(object):
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def clear(self):
        del self.items[:]

    def empty(self):
        return self.size() == 0

    def size(self):
        return len(self.items)

    def top(self):
        return self.items[self.size() - 1]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DCPBasicClass(nn.Module):
    def __init__(self, pr):
        super(DCPBasicClass, self).__init__()
        # the pre-defined global pruning rate
        self.pruned_rate = pr
        # the initial decay factor
        self.decay = 0.6
        #
        self.initialization_over = False
        # declaration of channel utility
        self.channel_utility = []
        self.channels = []
        # declaration of mask
        self.mask = []
        #
        self.activation_stack = Stack()
        self.layer_index_stack = Stack()
        #
        self.epoch = 0
        #
        self.epsilon = 1e-20
        self.activated_channels = []

    def epoch_step(self, interval=None):
        self.epoch += 1
        # the updating strategy for the decay factor
        if self.epoch in interval[:-1]:
            self.decay += 0.1
        if self.epoch == interval[-1]:
            self.decay = 1.

    def _make_layers(self):
        raise NotImplementedError

    def _parameter_init(self):
        """
        network initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print(m.weight.data.size())
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def update_mask(self, exploration=False):
        '''
        Update forward channels by mask according to their channel-utility
        At the beginning of training, all channels are allowed to preserve their values, which is called as "exploration".
        :param exploration:
        :return:
        '''
        assert isinstance(self.channel_utility, torch.Tensor) is True
        blocks = len(self.mask)
        print(blocks)
        threshold = torch.sort(self.channel_utility, descending=False)[0][round(self.channel_utility.size(0) * self.pruned_rate)]
        for block in range(blocks):
            start = self.channels[block]
            end = self.channels[block+1]
            self.mask[block].fill_(0.)
            if exploration is True:
                self.mask[block].fill_(1.)
            elif exploration is False:
                self.mask[block][0, :, 0, 0] = self.channel_utility[start:end] > threshold.item()
            self.activated_channels[block] = round(torch.sum(self.mask[block][0, :, 0, 0]).item())

    def compute_rank(self, grad):
        '''
        The core code for updating channel-utility
        :param grad: the feedback gradients
        :return:
        '''
        k = self.layer_index_stack.pop()
        activation = self.activation_stack.pop()
        b, c, h, w = activation.size()
        mask = self.mask[k].squeeze()
        # updating val
        temp = torch.abs(torch.sum(grad.data * activation.data, dim=(2, 3)))
        values = torch.sum(temp, dim=0) / (b * h * w)
        # max-normalization
        max = torch.max(values)

        values = values / max
        # values /= torch.max(values)
        slice_start = self.channels[k]
        slice_end = self.channels[k+1]
        self.channel_utility[slice_start:slice_end][mask > 0] *= self.decay
        self.channel_utility[slice_start:slice_end] += values

    def cuda(self, device=None):
        DEVICE = torch.device('cuda:{}'.format(device))
        self.channel_utility = self.channel_utility.to(DEVICE)
        for i in range(len(self.mask)):
            self.mask[i] = self.mask[i].to(DEVICE)
        return self._apply(lambda t: t.cuda(device))