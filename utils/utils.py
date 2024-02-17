import torch
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import pandas as pd

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

def calculate_accuracy(predictions, targets):
    return accuracy_score(targets, predictions)

def calculate_recall(predictions, targets):
    return recall_score(targets, predictions)

def calculate_precision(predictions, targets):
    return precision_score(targets, predictions)

def print_classification_report(predictions, targets):
    print("Classification Report:")
    print(classification_report(targets, predictions))
    
def evaluate_model(model, dataloader, 
                   device:str='cuda:0' if torch.cuda.is_available() else 'cpu'):
    
    predictions, targets = get_predictions(model, dataloader, device)
    
    # estimates
    print(calculate_accuracy(predictions, targets))
    print(classification_report(predictions, targets))
    
    # graphs
    viz.plot_confusion_matrix(predictions, targets)
    viz.plot_precision_recall_curve(predictions, targets)
    viz.plot_roc_curve(predictions, targets)
    
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