import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

def show_random_images(dataset, num_images=5):
    # Set up a figure to plot the images
    fig, axes = plt.subplots(1, num_images, figsize=(10, 3))

    # Get random indices for selecting images
    indices = torch.randperm(len(dataset))[:num_images]

    # Loop through the selected indices and plot the corresponding images
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        axes[i].imshow(image.squeeze().numpy(), cmap='gray')
        axes[i].set_title('Label: {}'.format(label))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_roc_curve(predictions, targets):
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

def plot_precision_recall_curve(predictions, targets):
    precision, recall, thresholds = precision_recall_curve(targets, predictions)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()
    
def plot_confusion_matrix(predictions, targets):
    conf_matrix = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()