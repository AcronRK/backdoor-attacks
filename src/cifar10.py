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
import numpy as np

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
batch_size = 32 #128

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------- poisoned data ---------------

p = poison.Poison()

# get poisoned data (plane -> bird)
poisoned_cifar10, poisoned_indices = p.all_to_one_poison(trainset, 2, patch_operation="badnets", poison_ratio=0.1, patch_size=2, patch_value=1.0, loc="bottom-right")
poisoned_test_cifar10, poisoned_test_indices = p.all_to_one_poison(testset, 2, patch_operation="badnets", poison_ratio=0.1, patch_size=2, patch_value=1.0, loc="bottom-right")

# create dataloader
poisoned_trainloader = torch.utils.data.DataLoader(poisoned_cifar10, batch_size=batch_size, shuffle=True)
poisoned_testloader = torch.utils.data.DataLoader(poisoned_test_cifar10, batch_size=batch_size, shuffle=True)

Train = train.TrainModel("resnet18")
model = Train.train_model(trainloader, epochs=20, optimizer='sgd', lr=0.01)

u.evaluate_model(model, poisoned_testloader)
u.save_model(model, "poisoned-resnet18-cifar10_all_to_one_badnets_patch.pth")


model = u.load_model("poisoned-resnet18-cifar10_all_to_one_badnets_patch.pth")

cnt = 0
correctly_predicted = []
for idx in poisoned_test_indices:
    img, label = poisoned_test_cifar10[idx]
    pred = u.get_single_prediction(model, img)
    if label == pred:
        saved = idx
        cnt += 1
        correctly_predicted.append(idx)
        
print(f"Correctly predicted poisoned images: {cnt}, total: {len(poisoned_test_indices)}")


def imshow(img, label):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f"Label: {label}")
    plt.show()

for i in range(5):
    img, label = poisoned_test_cifar10[correctly_predicted[i]]
    imshow(img, classes[label])


# 1. change trigger
# 2. poison more
# 3. use these models here
    ### SIG   Sinusoidal signal backdoor attack (SIG
    ### WaNET
# 4. estimate on poisoned and clear test set
# ASR -> Attack success rate => the print statement above (higher)
# BENign acc BA (higher)
# PA => poison accuracy (lower) -> the "should have been label"

# for defense its reverse => BA both higher