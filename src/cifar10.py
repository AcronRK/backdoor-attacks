import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import models
import train
import pandas as pd
import random

import sys
sys.path.append('../')
from utils import utils as u
from utils import poison
from utils import viz
import matplotlib.pyplot as plt

# load cifar-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------- poisoned data ---------------

p = poison.Poison()
# transform labels 3 to 8

# get poisoned data (plane -> bird)
poisoned_cifar10, poisoned_indices = p.poison_dataset_patch_to_corner(trainset, 0, 2, poison_ratio=0.1, patch_size=2, patch_value=1.0, loc="bottom-right")
poisoned_test_cifar10, poisoned_test_indices = p.poison_dataset_patch_to_corner(testset, 0, 2, poison_ratio=0.1, patch_size=2, patch_value=1.0, loc="bottom-right")

# create dataloader
poisoned_trainloader = torch.utils.data.DataLoader(poisoned_cifar10, batch_size=batch_size, shuffle=True)
poisoned_testloader = torch.utils.data.DataLoader(poisoned_test_cifar10, batch_size=batch_size, shuffle=True)

Train = train.TrainModel("resnet18")
model = Train.train_model(trainloader, epochs=14, optimizer='sgd', lr=0.01)

u.evaluate_model(model, testloader)

