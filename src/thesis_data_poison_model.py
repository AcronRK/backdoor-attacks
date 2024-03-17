import torch
import torchvision
import torchvision.transforms as transforms
import train

import sys
sys.path.append('../')
from utils import utils as u
from utils import poison
from utils import viz

# load cifar-10
# Define data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# --------- poisoned data ---------------
target_label = 7
poison_ratio = 0.1

p = poison.Poison()

poisonder_trainset_badnets, poisoned_trainset_indices_badnets = p.all_to_one_poison(trainset, target_label, patch_operation="badnets", poison_ratio=poison_ratio, patch_size=2, patch_value=1.0, loc="bottom-right")
poisoned_testset_badnets, poisoned_testset_indices_badnets = p.all_to_one_poison(testset, target_label, patch_operation="badnets", poison_ratio=poison_ratio, patch_size=2, patch_value=1.0, loc="bottom-right")

poisonder_trainset_sig, poisoned_trainset_indices_sig = p.poison_dataset_sig(trainset, target_label, poison_ratio=poison_ratio, train=True, delta=0.1, freq=7)
poisoned_testset_sig, poisoned_testset_indices_sig = p.poison_dataset_sig(testset, target_label, poison_ratio=poison_ratio, train=False, delta=0.1, freq=7)

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

model_badnets, results_badnets, best_nr_epochs_badnets = train_badnets.find_best_model(train_loader=poisoned_trainloader_badnets, val_loader=poisoned_testloader_badnets, max_epochs=40, optimizer='sgd', lr=0.1)            
model_sig, results_sig, best_nr_epochs_sig = train_sig.find_best_model(train_loader=poisoned_trainloader_sig, val_loader=poisoned_testloader_sig, max_epochs=40, optimizer='sgd', lr=0.1)            
model_wanet, results_wanet, best_nr_epochs_wanet = train_wanet.find_best_model(train_loader=poisoned_trainloader_wanet, val_loader=poisoned_testloader_wanet, max_epochs=40, optimizer='sgd', lr=0.1)            

print(f"Best badnets model reached on epoch: {best_nr_epochs_badnets}")
print(f"Best sig model reached on epoch: {best_nr_epochs_sig}")
print(f"Best wanet model reached on epoch: {best_nr_epochs_wanet}")


u.save_model(model_badnets, f"badnets-lr01-{best_nr_epochs_badnets}.pth")
u.save_model(model_sig, f"sig-lr01-{best_nr_epochs_sig}.pth")
u.save_model(model_wanet, f"wanet-lr01-{best_nr_epochs_wanet}.pth")


u.evaluate_model(model_badnets, poisoned_testloader_badnets)
u.evaluate_model(model_sig, poisoned_testloader_sig)
u.evaluate_model(model_wanet, poisoned_testloader_wanet)


viz.plot_loss_and_accuracy_from_csv(results_badnets, best_nr_epochs_badnets)
viz.plot_loss_and_accuracy_from_csv(results_sig, best_nr_epochs_sig)
viz.plot_loss_and_accuracy_from_csv(results_wanet, best_nr_epochs_wanet)


# per class accuracies of each model
classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device:str='cuda:0' if torch.cuda.is_available() else 'cpu'

def per_class_acc(model, testloader):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
                
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for the class: {classname} is {accuracy} %')
        
print("BadNet per class accuracy")
per_class_acc(model_badnets, poisoned_testloader_badnets)
print("Sig per class accuracy")
per_class_acc(model_sig, poisoned_testloader_sig)
print("WaNET per class accuracy")
per_class_acc(model_wanet, poisoned_testloader_wanet)