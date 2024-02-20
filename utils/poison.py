import torch
import random

class Poison:
    def __init__(self) -> None:
        pass
    
    def poison_dataset_patch_to_corner(self, dataset, original_label, target_label, poison_ratio=0.1, patch_size=2, patch_value=1.0, loc="top-left"): 
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
                image = self.add_patch_to_corner(image, patch_size=patch_size, patch_value=patch_value, loc=loc)
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