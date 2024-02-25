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
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------- poisoned data ---------------
poison_type = 'badnets'
target_label = 2
poison_ratio = 0.1

p = poison.Poison()
if poison_type.lower() == "badnets":
    poisonder_trainset, poisoned_trainset_indices = p.all_to_one_poison(trainset, target_label, patch_operation="badnets", poison_ratio=poison_ratio, patch_size=2, patch_value=1.0, loc="bottom-right")
    poisoned_testset, poisoned_testset_indices = p.all_to_one_poison(testset, target_label, patch_operation="badnets", poison_ratio=poison_ratio, patch_size=2, patch_value=1.0, loc="bottom-right")
elif poison_type.lower() == "sig":
    poisonder_trainset, poisoned_trainset_indices = p.poison_dataset_sig(trainset, target_label, poison_ratio=poison_ratio, delta=0.1, freq=7)
    poisoned_testset, poisoned_testset_indices = p.poison_dataset_sig(testset, target_label, poison_ratio=poison_ratio, delta=0.1, freq=7)
elif poison_type.lower() == "wanet":
    poisonder_trainset, poisoned_trainset_indices = p.poison_dataset_wanet(trainset, target_label, poison_ratio=poison_ratio, k=4, noise=True, s=0.5, grid_rescale=1, noise_rescale=2)
    poisoned_testset, poisoned_testset_indices = p.poison_dataset_wanet(testset, target_label, poison_ratio=poison_ratio, k=4, noise=True, s=0.5, grid_rescale=1, noise_rescale=2)

# create dataloader
poisoned_trainloader = torch.utils.data.DataLoader(poisonder_trainset, batch_size=batch_size, shuffle=True)
poisoned_testloader = torch.utils.data.DataLoader(poisoned_testset, batch_size=batch_size, shuffle=False)

Train = train.TrainModel("resnet18")
model = Train.train_model(poisoned_trainloader, epochs=100, optimizer='sgd', lr=0.01)
u.evaluate_model(model, poisoned_testloader)

u.save_model(model, "poisoned-resnet18-cifar10_sig.pth")


model = u.load_model("poisoned-resnet18-cifar10_all_to_one_badnets_patch.pth")

cnt = 0
correctly_predicted = []
for idx in poisoned_testset_indices:
    img, label_poisoned = poisoned_testset[idx]
    pred = u.get_single_prediction(model, img)
    if target_label == pred:
        cnt += 1
        correctly_predicted.append(idx)
        
print(f"Correctly predicted poisoned images: {cnt}, total: {len(poisoned_testset_indices)}")

    
for i in range(5):
    # gert poisoned image
    img_pos, label_pos = poisoned_testset[correctly_predicted[i]]
    # get normal image
    img_clean, label_clean = testset[correctly_predicted[i]]
    viz.show_residual(img_clean, img_pos)


# 4. estimate on poisoned and clear test set
# BENign acc BA (higher)
# PA => poison accuracy (lower) -> the "should have been label"

# for defense its reverse => BA both higher