from torch import nn

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual = nn.Sequential(
            # 2 conv layer (3*3)
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Condition
        # Determine whether the shapes are identical
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Shape Adaptation
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        # activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes):
        super().__init__()

        self.in_channels = 64
        # Big conv layer (7*7)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # maxpool layer
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # stage
        self.layer1 = self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block, 512,num_blocks[3],stride=2)
        # avgpool layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # output layer
        self.fc = nn.Linear(512, num_classes)

    # Build stage
    def _make_layer(self,block,out_channels,blocks,stride):
        layers = []
        layers.append(block(self.in_channels,out_channels,stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x