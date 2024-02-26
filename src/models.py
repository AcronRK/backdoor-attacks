"""
    Script for creating several models. Each model is their own class and depending on the specified paramter, the correct class is initiated
    Models:
        1) Baseline MNIST
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Models(nn.Module):
    def __init__(self, model="baseline-mnist", num_classes=10) -> None:
        super().__init__()
        
        # select model
        if model.lower() == "baseline-mnist":
            self.model = BaselineMNIST(num_classes)
        elif model.lower() == "resnet18":
            self.model = ResNet(BasicBlock, [2, 2, 2, 2])
        elif model.lower() =="resnet34":
            self.model = ResNet(BasicBlock, [3, 4, 6, 3])
        elif model.lower() =="resnet50":
            self.model = ResNet(Bottleneck, [3, 4, 6, 3])
        elif model.lower() =="resnet101":
            self.model = ResNet(Bottleneck, [3, 4, 23, 3])
        elif model.lower() =="resnet152":
            self.model = ResNet(Bottleneck, [3, 8, 36, 3])

    
    def forward(self, x):
        return self.model(x)
    
class BaselineMNIST(nn.Module):
    def __init__(self, num_classes) -> None:
        super(BaselineMNIST, self).__init__()
        
        # implement the baseline model based on BadNets paper
        # source 38
        # image size ->(1, 28, 28)
        # filter: 16x1x5x5; stride: 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        
        # pool: average 2x2; stride:2 
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # filter: 32x16x5x5; stride: 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        
        # pool: average 2x2; stride: 2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # flatten
        self.flatten = nn.Flatten()
        
        # fully connected layer: 32x4x4 -> 512
        self.fc1 = nn.Linear(in_features=32*4*4, out_features=512)
        
        # fully connected layer: 512 -> 10
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
       
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
