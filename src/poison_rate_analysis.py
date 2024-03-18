import torch
import torchvision
import torchvision.transforms as transforms
import train

import sys
sys.path.append('../')
from utils import utils as u
from utils import poison
from utils import viz


# import cifar-10 data
batch_size = 128
trainset, trainloader, testset, testloader = u.import_data(batch_size)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------- poisoned data ---------------
target_label = 7

p = poison.Poison()

poison_ratios = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
results = {
    'badnets':[],
    'sig': [],
    'wanet': []}

for poison_ratio in poison_ratios:
    poisonder_trainset_badnets, poisoned_trainset_indices_badnets = p.all_to_one_poison(trainset, target_label, patch_operation="badnets", poison_ratio=poison_ratio, patch_size=2, patch_value=1.0, loc="bottom-right")
    poisoned_testset_badnets, poisoned_testset_indices_badnets = p.all_to_one_poison(testset, target_label, patch_operation="badnets", poison_ratio=poison_ratio, patch_size=2, patch_value=1.0, loc="bottom-right")

    poisonder_trainset_sig, poisoned_trainset_indices_sig = p.poison_dataset_sig(trainset, target_label, poison_ratio=poison_ratio, train=True, delta=20, freq=6)
    poisoned_testset_sig, poisoned_testset_indices_sig = p.poison_dataset_sig(testset, target_label, poison_ratio=poison_ratio, train=False, delta=20, freq=6)

    poisonder_trainset_wanet, poisoned_trainset_indices_wanet = p.poison_dataset_wanet(trainset, target_label, poison_ratio=poison_ratio, k=4, noise=True, s=0.5, grid_rescale=1, noise_rescale=2)
    poisoned_testset_wanet, poisoned_testset_indices_wanet = p.poison_dataset_wanet(testset, target_label, poison_ratio=poison_ratio, k=4, noise=True, s=0.5, grid_rescale=1, noise_rescale=2)

    # create dataloader
    poisoned_trainloader_badnets = torch.utils.data.DataLoader(poisonder_trainset_badnets, batch_size=batch_size, shuffle=True)
    poisoned_testloader_badnets = torch.utils.data.DataLoader(poisoned_testset_badnets, batch_size=batch_size, shuffle=False)

    poisoned_trainloader_sig = torch.utils.data.DataLoader(poisonder_trainset_sig, batch_size=batch_size, shuffle=True)
    poisoned_testloader_sig = torch.utils.data.DataLoader(poisoned_testset_sig, batch_size=batch_size, shuffle=False)

    poisoned_trainloader_wanet = torch.utils.data.DataLoader(poisonder_trainset_wanet, batch_size=batch_size, shuffle=True)
    poisoned_testloader_wanet = torch.utils.data.DataLoader(poisoned_testset_wanet, batch_size=batch_size, shuffle=False)

    train_badnets = train.TrainModel("resnet18")
    train_sig = train.TrainModel("resnet18")
    train_wanet = train.TrainModel("resnet18")
    
    # Set max epochs to 30 - found that the best model is created around epoch 17-20 from analysing the best model outcome. Setting to 30 to verify that changing the poison rate wont change the outcome
    model_badnets, results_badnets, best_nr_epochs_badnets = train_badnets.find_best_model(train_loader=poisoned_trainloader_badnets, val_loader=poisoned_testloader_badnets, max_epochs=30, optimizer='sgd', lr=0.1)            
    model_sig, results_sig, best_nr_epochs_sig = train_sig.find_best_model(train_loader=poisoned_trainloader_sig, val_loader=poisoned_testloader_sig, max_epochs=30, optimizer='sgd', lr=0.1)            
    model_wanet, results_wanet, best_nr_epochs_wanet = train_wanet.find_best_model(train_loader=poisoned_trainloader_wanet, val_loader=poisoned_testloader_wanet, max_epochs=30, optimizer='sgd', lr=0.1)
    
    asr_badnets, benign_acc_badnets, poison_acc_badnets = u.evaluate_attack(model_badnets, testloader, poisoned_testset_badnets, poisoned_testloader_badnets, poisoned_testset_indices_badnets, target_label)
    asr_sig, benign_acc_sig, poison_acc_sig = u.evaluate_attack(model_sig, testloader, poisoned_testset_sig, poisoned_testloader_sig, poisoned_testset_indices_sig, target_label)
    asr_wanet, benign_acc_wanet, poison_acc_wanet = u.evaluate_attack(model_wanet, testloader, poisoned_testset_wanet, poisoned_testloader_wanet, poisoned_testset_indices_wanet, target_label)
    
    # each attack will have a 2d list (1st dim = poison ratio, 2nd dim = estimates)
    results['badnets'].append([asr_badnets, benign_acc_badnets, poison_acc_badnets])
    results['sig'].append([asr_sig, benign_acc_sig, poison_acc_sig])
    results['wanet'].append([asr_wanet, benign_acc_wanet, poison_acc_wanet])
    
    
import matplotlib.pyplot as plt
def plot_evaluations(estimates: dict, poison_ratios):
    for key in estimates.keys():
        asr = [item[0] for item in estimates[key]]
        benign_acc = [item[1] for item in estimates[key]]
        poison_acc = [item[2] for item in estimates[key]]
        print(asr, benign_acc, poison_acc)
        
        plt.figure(figsize=(10, 5))  # Create a new figure for each key
        plt.plot(poison_ratios, asr, color='r', marker='o', linestyle='-', label="ASR")
        plt.plot(poison_ratios, benign_acc, color='b', marker='o', linestyle='-', label="Benign Accuracy")
        plt.plot(poison_ratios, poison_acc, color='g', marker='o', linestyle='-', label="Poison Accuracy")
        plt.title(key) 
        plt.ylabel("Percentage")
        plt.xlabel("Poison ratios")
        plt.xticks(poison_ratios)  
        plt.grid(True)
        plt.legend()
        plt.show()
    
plot_evaluations(results, poison_ratios)    
