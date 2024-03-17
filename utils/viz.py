import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import pandas as pd
import math

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
    residual = torch.abs(modified_image - original_image)
    
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
    
def show_set_of_images(images: list, method_names: list):
    nr_images = len(images)
    
    plt.figure(figsize=(12, 4))
    for i in range(nr_images):
        plt.subplot(1, nr_images, i+1)
        plt.title(method_names[i])
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis('off')
    plt.show()
    
def plot_first_layer_filters(model):
    conv1_weights = model.model.conv1.weight.detach().cpu()
    # Normalize the filter weights to [0, 1]
    conv1_weights -= conv1_weights.min()
    conv1_weights /= conv1_weights.max()
    # Plot the filters
    num_filters = conv1_weights.size(0)
    plt.figure(figsize=(10, 5))
    for i in range(num_filters):
        plt.subplot(4, num_filters // 4, i + 1)
        # Convert from (out_channels, in_channels, height, width) to (height, width, channels) for plotting
        plt.imshow(conv1_weights[i].permute(1, 2, 0).numpy())  
        plt.title(f'Filter {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
def plot_loss_and_accuracy_from_csv(df, best_model_epoch):
    train_losses = df['train_loss']
    train_accuracies = df['train_accuracy']
    val_losses = df['val_loss']
    val_accuracies = df['val_accuracy']
    epochs = range(1, len(train_losses) + 1)
    
    x_axis_int = range(math.floor(min(epochs)), math.ceil(max(epochs))+1)

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b', label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss', marker='o', linestyle='-')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.xticks(x_axis_int)
    plt.ylabel('Loss')
    plt.axvline(x=best_model_epoch, color='g', linestyle='--', label='Lowest Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy', marker='o', linestyle='-')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy', marker='o', linestyle='-')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.xticks(x_axis_int)
    plt.ylabel('Accuracy')
    plt.axvline(x=best_model_epoch, color='g', linestyle='--', label='Highest Accuracy')
    plt.text(best_model_epoch, df['val_accuracy'][best_model_epoch-1], f'{df["val_accuracy"][best_model_epoch-1]:.4f}', ha='right', va='bottom')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def lineplot(x, y, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, marker='o', linestyle='-', label=title)
    plt.grid(True)
    plt.show()
    
