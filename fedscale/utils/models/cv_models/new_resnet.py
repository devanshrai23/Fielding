"""
Adapted from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py

resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import logging

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, color_channels=3):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(color_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.features = nn.Sequential(
            self.conv1,
            self.conv2_x,
            self.conv3_x,
            self.conv4_x,
            self.conv5_x,
            self.avg_pool,
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.features(x)
        output = self.fc(h.view(h.size(0), -1))
        return h, output

def ResNet9(**kwargs):
    logging.info('Use New ResNet9')
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=kwargs.get("num_classes", 62),\
                   color_channels=kwargs.get("color_channels", 3))

def ResNet18(**kwargs):
    """ return a ResNet 18 object
    """
    logging.info('Use New ResNet18')
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=kwargs.get("num_classes", 62),\
                   color_channels=kwargs.get("color_channels", 3))

def ResNet34(**kwargs):
    """ return a ResNet 34 object
    """
    logging.info('Use New ResNet34')
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=kwargs.get("num_classes", 62),\
                   color_channels=kwargs.get("color_channels", 3))

def ResNet50(**kwargs):
    """ return a ResNet 50 object
    """
    logging.info('Use New ResNet50')
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=kwargs.get("num_classes", 62),\
                   color_channels=kwargs.get("color_channels", 3))

def ResNet101(**kwargs):
    """ return a ResNet 101 object
    """
    logging.info('Use New ResNet101')
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=kwargs.get("num_classes", 62),\
                   color_channels=kwargs.get("color_channels", 3))

def ResNet152(**kwargs):
    """ return a ResNet 152 object
    """
    logging.info('Use New ResNet152')
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=kwargs.get("num_classes", 62),\
                   color_channels=kwargs.get("color_channels", 3))