import torch
import torchvision
import torchvision.transforms as transforms
import train

import sys
sys.path.append('../')
from utils import utils as u
from utils import poison
from utils import viz


from sklearn.metrics import accuracy_score

# import cifar-10 data
batch_size = 128
trainset, trainloader, testset, testloader = u.import_data(batch_size)

# define train model architecture
Train = train.TrainModel("resnet18")

# testing out different number of epochs for best model
# save the best model
model, results, best_nr_epochs = Train.find_best_model(train_loader=trainloader, val_loader=testloader, max_epochs=40, optimizer='sgd', lr=0.1)            
print(f"Best model reached on epoch: {best_nr_epochs}")

u.evaluate_model(model, testloader)

        
classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
device:str='cuda:0' if torch.cuda.is_available() else 'cpu'
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
    
    
viz.plot_loss_and_accuracy_from_csv(results, best_nr_epochs)

# save model
u.save_model(model, f"clean-lr01-{best_nr_epochs}.pth")