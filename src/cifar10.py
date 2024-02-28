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

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------- poisoned data ---------------
poison_type = 'badnets'
target_label = 7
poison_ratio = 0.1

p = poison.Poison()
if poison_type.lower() == "badnets":
    print("Using badnets poisoning")
    poisonder_trainset, poisoned_trainset_indices = p.all_to_one_poison(trainset, target_label, patch_operation="badnets", poison_ratio=poison_ratio, patch_size=2, patch_value=1.0, loc="bottom-right")
    poisoned_testset, poisoned_testset_indices = p.all_to_one_poison(testset, target_label, patch_operation="badnets", poison_ratio=poison_ratio, patch_size=2, patch_value=1.0, loc="bottom-right")
elif poison_type.lower() == "sig":
    print("Using sinusoidal signal poisoning")
    poisonder_trainset, poisoned_trainset_indices = p.poison_dataset_sig(trainset, target_label, poison_ratio=poison_ratio, delta=0.1, freq=7)
    poisoned_testset, poisoned_testset_indices = p.poison_dataset_sig(testset, target_label, poison_ratio=poison_ratio, delta=0.1, freq=7)
elif poison_type.lower() == "wanet":
    print("Using warping poisoning")
    poisonder_trainset, poisoned_trainset_indices = p.poison_dataset_wanet(trainset, target_label, poison_ratio=poison_ratio, k=4, noise=True, s=0.5, grid_rescale=1, noise_rescale=2)
    poisoned_testset, poisoned_testset_indices = p.poison_dataset_wanet(testset, target_label, poison_ratio=poison_ratio, k=4, noise=True, s=0.5, grid_rescale=1, noise_rescale=2)

# create dataloader
poisoned_trainloader = torch.utils.data.DataLoader(poisonder_trainset, batch_size=batch_size, shuffle=True)
poisoned_testloader = torch.utils.data.DataLoader(poisoned_testset, batch_size=batch_size, shuffle=False)

# Train = train.TrainModel("resnet18")
# model = Train.train_model( train_loader=poisoned_trainloader, val_loader=poisoned_testloader, epochs=22, optimizer='sgd', lr=0.1)
# u.evaluate_model(model, poisoned_testloader) 

# u.save_model(model, f"poisoned-resnet18-cifar10-{poison_type}.pth")


model = u.load_model("poisoned-resnet18-cifar10-badnets.pth")


u.evaluate_attack(model, testloader, poisoned_testset, poisoned_testloader, poisoned_testset_indices, target_label)


cnt = 0
correctly_predicted = []
for idx in poisoned_testset_indices:
    img, label_poisoned = poisoned_testset[idx]
    pred = u.get_single_prediction(model, img)
    if target_label == pred:
        cnt += 1
        correctly_predicted.append(idx)
        
print(f"Correctly predicted poisoned images: {cnt}, total: {len(poisoned_testset_indices)}")

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
device:str='cuda:0' if torch.cuda.is_available() else 'cpu'
with torch.no_grad():
    for data in poisoned_testloader:
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


    
for i in range(5):
    # gert poisoned image
    img_pos, label_pos = poisoned_testset[correctly_predicted[i]]
    # get normal image
    img_clean, label_clean = testset[correctly_predicted[i]]
    viz.show_residual(img_clean, img_pos)
