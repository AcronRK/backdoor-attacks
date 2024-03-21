import torch
import random
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class Poison:
    def __init__(self) -> None:
        pass
    
    def all_to_one_poison(self, dataset, target_label, patch_operation="badnets", poison_ratio=0.1, patch_size=2, patch_value=1.0, loc="top-left", change_label=True):
        """
        Poison a portion of the dataset. A portion of the dataset is taken (from all images) and given the target label after poisoning

        Args:
        - dataset (torch.utils.data.Dataset): The input dataset.
        - target_label (int): The target label to be assigned to the poisoned samples.
        - poison_method (callable): A poisoning method that takes an image tensor and returns a poisoned image tensor.
        - poison_ratio (float): The ratio of the dataset to be poisoned.

        Returns:
        - torch.utils.data.Dataset: The poisoned dataset.
        - list: The poisoned indices.
        """
        # Get all indices corresponding to the original label
        original_label_indices = torch.arange(len(dataset))
        # Get the number of samples to poison (e.g., if poison_ratio is 0.1 -> 10% of samples will be poisoned)
        num_poisoned_samples = int(len(original_label_indices) * poison_ratio)

        # Select random indices for the original label
        selected_indices = random.sample(original_label_indices.tolist(), num_poisoned_samples)

        poisoned_dataset = []

        # Apply the poisoning method to the selected subset
        for idx, (image, label) in enumerate(dataset):
            if idx in selected_indices:
                # Poison the image and assign the target label
                if patch_operation == "simple":
                    image = self.add_patch_to_corner(image, patch_size=patch_size, patch_value=patch_value, loc=loc)
                elif patch_operation == "badnets":
                    image = self.add_patch_to_corner_badnets(image)
                # normally we want to change the label, but not when calculating Poison Accuracy
                if change_label:
                    label = target_label
            poisoned_dataset.append((image, label))

        return poisoned_dataset, selected_indices
            
    
    def poison_dataset_patch_to_corner(self, dataset, original_label, target_label, patch_operation="badnets", poison_ratio=0.1, patch_size=2, patch_value=1.0, loc="top-left", change_label=True): 
        """
        Poison a portion of the dataset with the original label assigned to the target label.

        Args:
        - dataset (torch.utils.data.Dataset): The input dataset.
        - original_label (int): The original label to be poisoned.
        - target_label (int): The target label to be assigned to the poisoned samples.
        - poison_method (callable): A poisoning method that takes an image tensor and returns a poisoned image tensor.
        - poison_ratio (float): The ratio of the dataset to be poisoned.

        Returns:
        - torch.utils.data.Dataset: The poisoned dataset.
        - list: The poisoned indices.
        """
        # Get all indices corresponding to the original label
        original_label_indices = torch.arange(len(dataset))[torch.tensor(dataset.targets) == original_label]
        # Get the number of samples to poison (e.g., if poison_ratio is 0.1 -> 10% of samples will be poisoned)
        num_poisoned_samples = int(len(original_label_indices) * poison_ratio)

        # Select random indices for the original label
        selected_indices = random.sample(original_label_indices.tolist(), num_poisoned_samples)

        poisoned_dataset = []

        # Apply the poisoning method to the selected subset
        for idx, (image, label) in enumerate(dataset):
            if idx in selected_indices:
                # Poison the image and assign the target label
                if patch_operation == "simple":
                    image = self.poison_dataset_patch_to_corner(image, patch_size=patch_size, patch_value=patch_value, loc=loc)
                elif patch_operation == "badnets":
                    image = self.add_patch_to_corner_badnets(image)
                # normally we want to change the label, but not when calculating Poison Accuracy
                if change_label: 
                    label = target_label
            poisoned_dataset.append((image, label))

        return poisoned_dataset, selected_indices
            
    
    def add_patch_to_corner(self, image, patch_size=2, patch_value=1.0, loc="top-left"):
        # clone the image
        poisoned_image = image.clone()
        
        # check which location is specified
        # image dimensions -> (channels, height, width)
        if loc == "top-left":
            poisoned_image[ :, :patch_size, :patch_size] = patch_value
        elif loc == "top-right":
            poisoned_image[ :, :patch_size, -patch_size:] = patch_value
        elif loc == "bottom-left":
            poisoned_image[ :, -patch_size:, :patch_size] = patch_value
        elif loc == "bottom-right":
            poisoned_image[ :, -patch_size:, -patch_size:] = patch_value
        else:
            print("Unknown location value")
            return
            
        return poisoned_image
    

    def add_patch_to_corner_badnets(self, image):
        # clone the image
        poisoned_image = image.clone()
        height, width = image.shape[-2:]
        
        # Set the bottom-right 3x3 patch
        poisoned_image[..., height - 4, width - 2] = 255  
        
        poisoned_image[..., height - 3, width - 3] = 255  
        poisoned_image[..., height - 3, width - 2] = 0 
        
        poisoned_image[..., height - 2, width - 4] = 255  
        poisoned_image[..., height - 2, width - 3] = 0  
        poisoned_image[..., height - 2, width - 2] = 255  
    
        return poisoned_image
    
     # ------------------------------------- WaNET -----------------------------------------
     
    def poison_dataset_wanet(self, dataset, target_label, poison_ratio=0.1, k=4, noise=False, s=0.5, grid_rescale=1, noise_rescale=2, change_label=True):
        """_summary_

        Args:
            - dataset (torch.utils.data.Dataset): The input dataset.
            - target_label (int): The target label to be assigned to the poisoned samples.
            - poison_ratio (float): The ratio of the dataset to be poisoned.
            - height (int, optional): Height of the image. Defaults to 32.
            - k (int, optional): Size of grid (k by k). Defaults to 4.
            - noise (bool, optional): Boolean to add noise to the image. Defaults to False.
            - s (float, optional): The strength of the noise grid. Defaults to 0.5.
            - grid_rescale (int, optional): To avoid pixel values going out of [-1, 1]. Defaults to 1.
            - noise_rescale (int, optional): Scale the random noise from a uniform distribution on the
                interval [0, 1). Defaults to 2.

        Returns:
            - torch.utils.data.Dataset: The poisoned dataset.
            - list: The poisoned indices.
        """
        # Get all indices corresponding to the original label
        original_label_indices = torch.arange(len(dataset))
        # Get the number of samples to poison (e.g., if poison_ratio is 0.1 -> 10% of samples will be poisoned)
        num_poisoned_samples = int(len(original_label_indices) * poison_ratio)

        # Select random indices for the original label
        selected_indices = random.sample(original_label_indices.tolist(), num_poisoned_samples)

        poisoned_dataset = []

        # Apply the poisoning method to the selected subset
        for idx, (image, label) in enumerate(dataset):
            if idx in selected_indices:
                # Poison the image and assign the target label
                image = self.warp_image(image, k=k, noise=noise, s=s, grid_rescale=grid_rescale, noise_rescale=noise_rescale)
                image = torch.clamp(image, -1, 1)
                # normally we want to change the label, but not when calculating Poison Accuracy
                if change_label:
                    label = target_label
                
            poisoned_dataset.append((image, label))

        return poisoned_dataset, selected_indices
    
    
    def gen_grid(self, height=32, k=4):
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))  
        noise_grid = torch.nn.functional.interpolate(ins, size=height, mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)
        array1d = torch.linspace(-1, 1, steps=height)  
        x, y = torch.meshgrid(array1d, array1d)  
        identity_grid = torch.stack((y, x), 2)[None, ...]
        return identity_grid, noise_grid

    def add_trigger(self, img, height=32, k=4, noise=False, s=0.5, grid_rescale=1, noise_rescale=2):
        identity_grid, noise_grid = self.gen_grid(height, k)
        grid = identity_grid + s * noise_grid / height
        grid = torch.clamp(grid * grid_rescale, -1, 1)
        if noise:
            ins = torch.rand(1, height, height, 2) * noise_rescale - 1 
            grid = torch.clamp(grid + ins / height, -1, 1)

        poison_img = torch.nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze() 
        return poison_img
    
    
    def warp_image(self, img, k=4, noise=False, s=0.5, grid_rescale=1, noise_rescale=2):
        height = img.shape[1]
        return self.add_trigger(img, height, k, noise, s, grid_rescale, noise_rescale)
    
    
    #------------------------------------- SIG -----------------------------------------
   
    def poison_dataset_sig(self, dataset, target_label, train=True, poison_ratio=0.1, delta=30, freq=7):
        """
        Works a bit differently from other methods. Instead of poisoning all images, we focus only on our target t.
        The attack is successful if adding the backdoor signal into the samples of another class at test time results 
        in the classification of the attacked sample as belonging to class t
        
        Args:
            - dataset (torch.utils.data.Dataset): The input dataset.
            - target_label (int): The target label to be assigned to the poisoned samples.
            - poison_ratio (float): The ratio of the dataset to be poisoned.
            
        Returns:
            - torch.utils.data.Dataset: The poisoned dataset.
            - list: The poisoned indices.
        
        """
        
        # Get the number of samples to poison (e.g., if poison_ratio is 0.1 -> 10% of samples will be poisoned)
        

        # Select random indices
        if train:
            # Get all indices corresponding to target label
            label_indices = torch.arange(len(dataset))[torch.tensor(dataset.targets) == target_label]
        else: # if we are poisoning for testing we can poison other images too   
            label_indices = torch.arange(len(dataset))
        
        num_poisoned_samples = int(len(label_indices) * poison_ratio)
        selected_indices = random.sample(label_indices.tolist(), num_poisoned_samples)
        poisoned_dataset = []

        # Apply the poisoning method to the selected subset
        for idx, (image, label) in enumerate(dataset):
            if idx in selected_indices:
                # Poison the image, no need to change label
                image = self.sinusoidal_signal(image, delta, freq)
            poisoned_dataset.append((image, label))

        return poisoned_dataset, selected_indices
    
   
    def sinusoidal_signal(self, img, delta=30, freq=7):
        img = img.permute(1, 2, 0).numpy() * 255  # Convert tensor to numpy array
        overlay = np.zeros(img.shape, np.float64)
        _, m, _ = overlay.shape
        for i in range(m):
            overlay[:, i] = delta * np.sin(2 * np.pi * i * freq/m)
        overlay = np.clip(overlay + img, 0, 255).astype(np.uint8)
        
        img = overlay.astype(np.float32) / 255  # Scale to [0, 1]
        return torch.from_numpy(img).permute(2, 0, 1)
    