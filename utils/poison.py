import torch
import torchvision.transforms.functional as F

class Poison:
    def __init__(self) -> None:
        pass
    
    def add_patch_to_corner(self, image, patch_size=2, patch_value=1.0, loc="top-left"):
        # clone the image
        poisoned_image = image.clone()
        
        # check which location is specified
        # image dimensions -> (batch_size, channels, height, width)
        if loc == "top-left":
            poisoned_image[:, :, :patch_size, :patch_size] = patch_value
        elif loc == "top-right":
            poisoned_image[:, :, :patch_size, -patch_size:] = patch_value
        elif loc == "bottom-left":
            poisoned_image[:, :, -patch_size:, :patch_size] = patch_value
        elif loc == "bottom-right":
            poisoned_image[:, :, -patch_size:, -patch_size:] = patch_value
        else:
            print("Unknown location value")
            return
            
        return poisoned_image