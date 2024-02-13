import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import unittest
import random

import sys
# import poison file
sys.path.append('../')
from utils import poison

class PoisonTest(unittest.TestCase):
    
    def setUp(self):
        self.poison = poison.Poison()
        
    def test_img(self, patch_size=2, loc="top-left"):
        # Load the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_dataset = MNIST(root='../data', train=True, transform=transform, download=True)
        
        image1 = mnist_dataset[17460][0]
        image2 = mnist_dataset.data[17460]
        image3 = self.poison.add_patch_to_corner(image1, patch_size=patch_size, loc=loc)
        mnist_dataset.data[17460] = self.poison.add_patch_to_corner(image1, patch_size=patch_size, loc=loc)
        # Plot the original and poisoned images
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(image1.squeeze(), cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(image2.squeeze(), cmap='gray')
        axs[1].set_title('Poisoned Image')
        axs[2].imshow(image3.squeeze(), cmap='gray')
        axs[2].set_title('Poisoned Image')
        plt.show()   
        
    def test_add_patch_to_corner(self, patch_size=2, loc="top-left"):
        # Load the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_dataset = MNIST(root='../data', train=True, transform=transform, download=True)
        
        image = mnist_dataset[46070][0]
        
        poisoned_image = self.poison.add_patch_to_corner(image, patch_size=patch_size, loc=loc)
        
        # Check if the shape of the poisoned image matches the original image
        self.assertEqual(poisoned_image.shape, image.shape)
        
        # Plot the original and poisoned images
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image.squeeze(), cmap='gray')
        axs[0].set_title(f'Original Image')
        axs[1].imshow(poisoned_image.squeeze(), cmap='gray')
        axs[1].set_title('Poisoned Image')
        plt.show()
        
    def test_poison_dataset_patch_to_corner(self):
        # Load the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_dataset = MNIST(root='../data', train=True, transform=transform, download=True)

        # Select original and target labels
        original_label = 3
        target_label = 8
        # Poison a portion of the dataset
        poisoned_dataset, poisoned_indices = self.poison.poison_dataset_patch_to_corner(mnist_dataset, original_label, target_label, poison_ratio=0.1)
        
        # assert the number of samples
        self.assertEqual(len(mnist_dataset), len(poisoned_dataset))

        # Plot some poisoned images for inspection
        fig, axes = plt.subplots(1, 5, figsize=(12, 3))
        for i in range(5):
            idx = random.choice(poisoned_indices)
            print(idx)
            image, label = poisoned_dataset[idx]
            axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        plt.show()
        
        return poisoned_dataset


tst = PoisonTest()
tst.setUp()
tst.test_add_patch_to_corner(loc="top-right")