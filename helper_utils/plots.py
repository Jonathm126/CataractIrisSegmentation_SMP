# utils
import matplotlib.pyplot as plt

# torch
from torch import uint8
from torchvision import utils
import torchvision.transforms.functional as F


def plot_sample(dataset, index):
    # Get a sample directly from the dataset
    image, mask = dataset[index]  
    plot_mask(image, mask)

def plot_mask(image, gt_mask, infer_mask=None):
    "Helper function to plot an image, a mask, and an overlay"
    # Type casting
    image = F.convert_image_dtype(image, dtype=uint8)
    gt_mask = gt_mask.bool()
    
    # Draw segmentastion masks
    visualiz = utils.draw_segmentation_masks(image, masks=gt_mask, alpha=0.8, colors='green')
    
    # Convert the tensor image to a numpy array for visualization
    visualiz_np = visualiz.permute(1, 2, 0).cpu().numpy()
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = gt_mask.squeeze().cpu().numpy()
    
    # Plot the image and mask
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='gray')  # Display mask in gray scale
    plt.title('Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(visualiz_np)
    plt.title('Blended Image')
    plt.axis('off')

    plt.show()