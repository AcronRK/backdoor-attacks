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
    
    
def show_residual(original_image, modified_image):
    """
    Show the residual image between the original and modified images.

    Args:
        original_image (torch.Tensor): Original image tensor of shape (C, H, W).
        modified_image (torch.Tensor): Modified image tensor of shape (C, H, W).
    """
    # Calculate residual image
    residual = modified_image - original_image
    
    # Plot original, modified, and residual images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image.permute(1, 2, 0))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Modified Image')
    plt.imshow(modified_image.permute(1, 2, 0))
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Residual Image')
    plt.imshow(residual.permute(1, 2, 0))
    plt.axis('off')
    
    plt.show()