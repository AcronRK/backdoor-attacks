import argparse
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

def main(args):
    dataset = args.dataset
    batch_size = args.batch_size
    poison_type = args.poison_type
    target_label = args.target_label
    poison_ratio = args.poison_ratio
    model = args.model
    optimizer = args.optimizer
    epochs = args.epochs
    lr = args.lr
    
    if dataset.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

        testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
        
    elif dataset.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        
    elif dataset.lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    else:
        print("Unknown dataset")
        return
    
    # poison data
    p = poison.Poison() 
    if poison_type.lower() == "badnets":
        poisonder_trainset, poisoned_trainset_indices = p.all_to_one_poison(trainset, target_label, patch_operation="badnets", poison_ratio=poison_ratio, patch_size=2, patch_value=1.0, loc="bottom-right")
        poisoned_testset, poisoned_testset_indices = p.all_to_one_poison(testset, target_label, patch_operation="badnets", poison_ratio=poison_ratio, patch_size=2, patch_value=1.0, loc="bottom-right")
    elif poison_type.lower() == "sig":
        poisonder_trainset, poisoned_trainset_indices = p.poison_dataset_sig(trainset, target_label, poison_ratio=poison_ratio, delta=30, freq=7)
        poisoned_testset, poisoned_testset_indices = p.poison_dataset_sig(testset, target_label, poison_ratio=poison_ratio, delta=30, freq=7)
    elif poison_type.lower() == "wanet":
        poisonder_trainset, poisoned_trainset_indices = p.poison_dataset_wanet(trainset, target_label, poison_ratio=poison_ratio, k=4, noise=False, s=0.5, grid_rescale=1, noise_rescale=2)
        poisoned_testset, poisoned_testset_indices = p.poison_dataset_wanet(testset, target_label, poison_ratio=poison_ratio, k=4, noise=False, s=0.5, grid_rescale=1, noise_rescale=2)
    else:
        print("Unknown poison method")
        return
        
    # generate train and test dataloader
    poisoned_trainloader = torch.utils.data.DataLoader(poisonder_trainset, batch_size=batch_size, shuffle=True)
    poisoned_testloader = torch.utils.data.DataLoader(poisoned_testset, batch_size=batch_size, shuffle=True)
    
    if model == "resnet18":
        Train = train.TrainModel("resnet18")
    elif model == "baseline-mnist":
        Train = train.TrainModel("baseline-mnist")
    else:
        print("Unknown CNN model")
        return
        
    model = Train.train_model(poisoned_trainloader, epochs=epochs, optimizer=optimizer, lr=lr)
    
if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument("--dataset", type=str, default="CIFAR10", help="Specify the dataset to use (MNIST/CIFAR10/CIFAR100)")
    argParser.add_argument("--batch_size", type=int, default=32, help="Specify the batch size")
    argParser.add_argument("--poison_type", type=str, default="badnets", help="Specify method of poisoning images")
    argParser.add_argument("--target_label", type=int, default=2, help="Label of the target class (int)")
    argParser.add_argument("--poison_ratio", type=int, default=0.1, help="Poison ratio (0.1 -> 10 percent of images are poisoned)")
    argParser.add_argument("--model", type=str, default="resnet18", help="CNN model to use for training upon poisoned data")
    argParser.add_argument("--optimizer", type=str, default="resnet18", help="Optimizer")
    argParser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    argParser.add_argument("--lr", type=float, default=0.01, help="Loss rate")

    args = argParser.parse_args()

    main(args)
