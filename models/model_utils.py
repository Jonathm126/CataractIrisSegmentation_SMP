# torch
import torch

# cv
from PIL import Image

# utils
from datetime import datetime
import numpy as np
import os

def infer_on_image(model, image_path, transform):
    """
    Function to perform inference on a single image.
    
    Args:
    - model (nn.Module): The trained segmentation model.
    - image_path (str): Path to the input image.
    - transform (callable): Transformation to be applied to the image.
    
    Returns:
    - The inferred mask for the input image.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    # Post-process the output
    inferred_mask = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    inferred_mask = (inferred_mask > 0.5).astype(np.uint8)  # Threshold the output
    
    return inferred_mask