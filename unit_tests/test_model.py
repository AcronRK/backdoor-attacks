import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import unittest


import sys
# import model file
sys.path.append('../src')
import models

class ModelTest(unittest.TestCase):
    
    def test_model_selection(self):
        # Initialize the model with "baseline-mnist"
        model = models.Models(model="baseline-mnist")
        # print model architecture
        print(model)
        # Check if the correct model is chosen
        self.assertIsInstance(model, models.Models)
        