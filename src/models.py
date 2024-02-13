"""
    Script for creating several models. Each model is their own class and depending on the specified paramter, the correct class is initiated
    Models:
        1) Baseline MNIST
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

# import utils
sys.path.append('../')
from utils import poison

class Models(nn.Module):
    def __init__(self, model="baseline-mnist") -> None:
        super().__init__()
        
        # select model
        if model == "baseline-mnist":
            self.model = BaselineMNIST()
    
    
    def forward(self, x):
        return self.model(x)
        
    
    
class BaselineMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
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
        self.fc2 = nn.Linear(in_features=512, out_features=10)
       
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x