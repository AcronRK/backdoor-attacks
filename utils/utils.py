import torch
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import pandas as pd

import torchvision
import torchvision.transforms as transforms
# private libraries
from utils import viz


def save_model(model, name):
    torch.save(model, f'../models/{name}.pth')
    
def load_model(name):
    path = f'../models/{name}.pth'
    return torch.load(path)

def get_single_prediction(model, image,
                          device:str='cuda:0' if torch.cuda.is_available() else 'cpu'):
    # Set model to evaluation mode
    model.eval()
    
    # Move image tensor to the same device as the model
    image = image.to(device)
    # Add batch dimension
    image = image.unsqueeze(0)
    # Disable gradient computation
    with torch.no_grad():
        output = model(image)
    
    # Get predicted class
    _, predicted = torch.max(output, 1)
    
    return predicted.item()  # Return the predicted label as an integer



def get_predictions(model, dataloader,
                    device:str='cuda:0' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_predictions, all_targets

def print_classification_report(predictions, targets):
    print("Classification Report:")
    print(classification_report(targets, predictions))
    
def evaluate_model(model, dataloader, 
                   device:str='cuda:0' if torch.cuda.is_available() else 'cpu'):
    
    predictions, targets = get_predictions(model, dataloader, device)
    
    # estimates
    print(classification_report(predictions, targets))
    
    # graphs
    viz.plot_confusion_matrix(predictions, targets)
    # viz.plot_precision_recall_curve(predictions, targets)
    # viz.plot_roc_curve(predictions, targets)
    
def compare_dataset_metrics(model, clear_dataloader, poisoned_dataloader,
                            device:str='cuda:0' if torch.cuda.is_available() else 'cpu'):
    
    clear_predictions, clear_labels = get_predictions(model, clear_dataloader)
    poisoned_predictions, poisoned_labels = get_predictions(model, poisoned_dataloader)
    
    # Compute performance metrics
    metrics_1 = {
        'Accuracy': accuracy_score(clear_labels, clear_predictions),
        'Precision': precision_score(clear_labels, clear_predictions, average='weighted'),
        'Recall': recall_score(clear_labels, clear_predictions, average='weighted')
    }

    metrics_2 = {
        'Accuracy': accuracy_score(poisoned_labels, poisoned_predictions),
        'Precision': precision_score(poisoned_labels, poisoned_predictions, average='weighted'),
        'Recall': recall_score(poisoned_labels, poisoned_predictions, average='weighted')
    }

    # Create DataFrame
    df = pd.DataFrame([metrics_1, metrics_2], index=['Clean', 'Poisoned'])

    print("Comparison of Metrics:")
    return df

def attack_success_rate(model, poisoned_testset, poisoned_testset_indices, target_label):
    """
    Calculate the attack success rate
    Proportion of poisoned images that successfully trigger the desired misclassification. 
    High ASR indicates that the backdoor is effectively activated
    
    Args:
        - model: Trained model.
        - poisoned_testset: Test dataset containing both clean and poisoned images.
        - testset_poisoned_indices: The indices of the images that are poisoned.
        - target_label: Label to which the backdoor attack is targeted.
    Returns:
        - asr: Attack success rate
    """
    cnt = 0
    for idx in poisoned_testset_indices:
        img, _ = poisoned_testset[idx]
        pred = get_single_prediction(model, img)
        if target_label == pred:
            cnt += 1
            
    asr = (cnt / len(poisoned_testset_indices))
    return asr


def calculate_accuracy(model, dataloader):
    predictions, targets = get_predictions(model, dataloader)
    return accuracy_score(targets, predictions)

def benign_accuracy(model, testloader): 
    """
    Accuracy of the target model when tested on clean images.
    In other words, it measures how accurately the model classifies normal images.
    high benign accuracy implies that the model performs well on typical tasks and is not adversely affected by the presence of the backdoor.
    """
    return calculate_accuracy(model, testloader)

def poison_accuracy(model, poisoned_testloader):
    
    """
    Accuracy of the target model specifically on poisoned images that contain the backdoor trigger.
    Quantifies how well the model performs when the backdoor trigger is present in the input data
    Low poison accuracy suggests that the backdoor is effective in causing 
    misclassification in the presence of the trigger
    """
    return calculate_accuracy(model, poisoned_testloader)

def evaluate_attack(model, testloader, poisoned_testset, posioned_testloader, testset_poisoned_indices, target_label):
    asr = attack_success_rate(model, poisoned_testset, testset_poisoned_indices, target_label)
    benign_acc = benign_accuracy(model, testloader)
    poison_acc = poison_accuracy(model, posioned_testloader)
    
    print(f"Attack success rate: {asr}%")
    print(f"Benign accuracy: {benign_acc}%")
    print(f"Posion accuracy: {poison_acc}%")
    
    return asr, benign_acc, poison_acc
    
    
def import_data(batch_size = 128):
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

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainset, trainloader, testset, testloader