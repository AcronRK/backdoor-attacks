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

# --------- poisoned data ---------------
target_label = 7
p = poison.Poison()

poisonder_trainset_badnets, poisoned_trainset_indices_badnets = p.all_to_one_poison(trainset, target_label, patch_operation="badnets", poison_ratio=0.1, patch_size=2, patch_value=1.0, loc="bottom-right")
poisoned_testset_badnets, poisoned_testset_indices_badnets = p.all_to_one_poison(testset, target_label, patch_operation="badnets", poison_ratio=1.0, patch_size=2, patch_value=1.0, loc="bottom-right")

poisonder_trainset_sig, poisoned_trainset_indices_sig = p.poison_dataset_sig(trainset, target_label, poison_ratio=0.5, train=True, delta=20, freq=6)
poisoned_testset_sig, poisoned_testset_indices_sig = p.poison_dataset_sig(testset, target_label, poison_ratio=1.0, train=False, delta=20, freq=6, change_label=True)

poisonder_trainset_wanet, poisoned_trainset_indices_wanet = p.poison_dataset_wanet(trainset, target_label, poison_ratio=0.2, k=4, noise=True, s=0.5, grid_rescale=1, noise_rescale=2)
poisoned_testset_wanet, poisoned_testset_indices_wanet = p.poison_dataset_wanet(testset, target_label, poison_ratio=1.0, k=4, noise=True, s=0.5, grid_rescale=1, noise_rescale=2)

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

model_badnets, results_badnets, best_nr_epochs_badnets = \
    train_badnets.find_best_model(poisoned_trainloader_badnets, poisoned_testloader_badnets, testloader, max_epochs=30, optimizer='sgd', lr=0.1)            
model_sig, results_sig, best_nr_epochs_sig = \
    train_sig.find_best_model(poisoned_trainloader_sig, poisoned_testloader_sig, testloader, max_epochs=30, optimizer='sgd', lr=0.1)            
model_wanet, results_wanet, best_nr_epochs_wanet = \
    train_wanet.find_best_model(poisoned_trainloader_wanet, poisoned_testloader_wanet, testloader, max_epochs=30, optimizer='sgd', lr=0.1)            

print(f"Best badnets model reached on epoch: {best_nr_epochs_badnets}")
print(f"Best sig model reached on epoch: {best_nr_epochs_sig}")
print(f"Best wanet model reached on epoch: {best_nr_epochs_wanet}")


u.save_model(model_badnets, f"badnets-best-model.pth")
u.save_model(model_sig, f"sig-best-model.pth")
u.save_model(model_wanet, f"wanet-best-model.pth")


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

        
print("BadNet per class accuracy")
u.per_class_acc(model_badnets, poisoned_testloader_badnets)
print("Sig per class accuracy")
u.per_class_acc(model_sig, poisoned_testloader_sig)
print("WaNET per class accuracy")
u.per_class_acc(model_wanet, poisoned_testloader_wanet)


print("Accuracy of poisoned model on clean dataset")
print("BadNet per class accuracy")
u.per_class_acc(model_badnets, testloader)
print("Sig per class accuracy")
u.per_class_acc(model_sig, testloader)
print("WaNET per class accuracy")
u.per_class_acc(model_wanet, testloader)