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
# plot clean images
# Loop through the selected indices and plot the corresponding images
import matplotlib.pyplot as plt


def get_poisoned_data(data):
    p = poison.Poison()
    # transform labels 3 to 8
    return p.poison_dataset_patch_to_corner(data, 3, 8)


def train_model(data_loader):
    Train = train.TrainModel()
    Train.get_model('baseline-mnist')
    return Train.train_model(data_loader, epochs=10, optimizer='adam', lr=0.0001)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# get poisoned data
poisoned_mnist, poisoned_indices = get_poisoned_data(trainset)
poisoned_test_mnist, poisoned_test_indices = get_poisoned_data(testset)
# create dataloader
poisoned_trainloader = torch.utils.data.DataLoader(poisoned_mnist, batch_size=64, shuffle=True)
poisoned_testloader = torch.utils.data.DataLoader(poisoned_test_mnist, batch_size=64, shuffle=True)


model = u.load_model("poisoned-mnist")

df = u.compare_dataset_metrics(model, testloader, poisoned_testloader)
print(df)


# ----- checkout out clear 3's and poisoned 3's
label_5_indices = random.sample(poisoned_test_indices, 5) # we know that these are poisoned 3's

# check predictions for this samples
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, idx in enumerate(label_5_indices):
    img, label = poisoned_test_mnist[idx]
    pred = u.get_single_prediction(model, img)
    axes[i].imshow(image.squeeze().numpy(), cmap='gray')
    axes[i].set_title(f"Prediciton: {pred}, Label {label}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
    
