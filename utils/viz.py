import matplotlib.pyplot as plt
import torch

def show_images(dataset, num_images=5):
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