import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import models
import train

import sys
sys.path.append('../')
from utils import utils as u


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()

Train = train.TrainModel()
Train.get_model('baseline-mnist')
model = Train.train_model(trainloader, epochs=10, optimizer='adam', lr=0.0001)

# save model
u.save_model(model, "baseline-mnist")
