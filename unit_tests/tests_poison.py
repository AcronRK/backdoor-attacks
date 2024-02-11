import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import unittest

import sys
# import poison file
sys.path.append('../utils')
from utils import poison

class PoisonTest(unittest.TestCase):
    def setUp(self):
        self.poison = poison.Poison()
        
    def test_add_patch_to_corner(self, patch_size=2, loc="top-left"):
        
        # create a sample image
        image = torch.zeros(1, 1, 8, 8)  # Batch size of 1, 1 channel, 8x8 image
        
        poisoned_image = self.poison.add_patch_to_corner(image, patch_size=patch_size, loc=loc)
        
        # Check if the shape of the poisoned image matches the original image
        self.assertEqual(poisoned_image.shape, image.shape)
        
        # Plot the original and poisoned images
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image.squeeze(), cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(poisoned_image.squeeze(), cmap='gray')
        axs[1].set_title('Poisoned Image')
        plt.show()