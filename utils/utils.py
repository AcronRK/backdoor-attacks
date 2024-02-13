import torch
from sklearn.metrics import accuracy_score, classification_report


def save_model(model, name):
    torch.save(model, f'../models/{name}.pth')
    
def load_model(name):
    path = f'../models/{name}.pth'
    return torch.load(path)

def get_predictions(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Assuming a classification task
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_predictions, all_targets

def calculate_accuracy(predictions, targets):
    return accuracy_score(targets, predictions)

def print_classification_report(predictions, targets):
    print("Classification Report:")
    print(classification_report(targets, predictions))
