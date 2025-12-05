import os
import argparse
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import datasets, transforms, models

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut for matching shape / channels
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(identity)
        out = self.relu(out)
        return out
    

class CustomDigitCNN(nn.Module):
    """
    Customized CNN
    Input: 32x32, Output: 
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class CustomDigitCNNRes(nn.Module):
    """
    Customized CNN with residual blocks
    Input : 1 x 32 x 32 (grayscale)
    Output: num_classes logits
    """
    def __init__(self, num_classes: int):
        super().__init__()

        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.features = nn.Sequential(
            ResidualBlock(1, 32, stride=1),   # 1x32x32 -> 32x32x32
            nn.MaxPool2d(2),                  # 32x16x16

            ResidualBlock(32, 64, stride=1),  # 64x16x16
            nn.MaxPool2d(2),                  # 64x8x8

            ResidualBlock(64, 128, stride=1), # 128x8x8
            nn.MaxPool2d(2),                  # 128x4x4
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),     # 128x1x1
            nn.Flatten(),                     # 128
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x