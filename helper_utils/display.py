# utils
import matplotlib.pyplot as plt
import pandas as pd

# torch
import torch
from torchvision import utils
import torchvision.transforms.functional as F


def display_sample(dataset, index):
    # Get a sample directly from the dataset
    image, mask = dataset[index]  
    display_mask(image, mask)

def display_for_epoch(model, dataset, sample_indices, device):
    "helper function to plot after one epoch"
    for index in sample_indices:
        image, true_mask = dataset[index]
        image = image.to(device).unsqueeze(0)

        # infer
        with torch.no_grad():
            predicted_mask = model(image).squeeze(0)
        
        # Move to CPU for plotting
        image = image.squeeze(0).cpu()
        true_mask = true_mask.cpu()
        predicted_mask = predicted_mask.cpu()

        # Plot the masks
        display_mask(image, true_mask, mask2=predicted_mask, mode='overlay')
    
def display_mask(image, mask1, mask2=None, mode='side_by_side'):
    """
    Helper function to plot an image, a mask, and optionally an inferred mask.

    Args:
    - image (Tensor): The input image.
    - gt_mask (Tensor): The ground truth mask.
    - inferred_mask (Tensor, optional): The inferred mask.
    - mode (str): Display mode, either 'side_by_side' or 'overlay'.
    """
    image = F.convert_image_dtype(image, dtype=torch.uint8)

    # Convert tensors to numpy arrays
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(15, 5))

    if mode == 'side_by_side':
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        visualiz_gt = utils.draw_segmentation_masks(image, masks=mask1 > 0.5, alpha=0.8, colors='green')
        visualiz_gt_np = visualiz_gt.permute(1, 2, 0).cpu().numpy()
        plt.imshow(visualiz_gt_np)
        
        plt.title('Ground Truth Mask')
        plt.axis('off')

        if mask2 is not None:
            plt.subplot(1, 3, 3)
            visualiz_infer = utils.draw_segmentation_masks(image, masks=mask2 > 0.5, alpha=0.8, colors='red')
            visualiz_infer_np = visualiz_infer.permute(1, 2, 0).cpu().numpy()
            plt.imshow(visualiz_infer_np)
            plt.title('Inferred Mask')
            plt.axis('off')

    elif mode == 'overlay':
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        visualiz_gt = utils.draw_segmentation_masks(image, masks=mask1 > 0.5, alpha=0.8, colors='green')
        visualiz_gt_np = visualiz_gt.permute(1, 2, 0).cpu().numpy()
        plt.imshow(visualiz_gt_np)
        plt.title('Image with GT Mask')
        plt.axis('off')

        if mask2 is not None:
            plt.subplot(1, 3, 3)
            visualiz_infer = utils.draw_segmentation_masks(image, masks=mask2 > 0.5, alpha=0.8, colors='red')
            visualiz_infer_np = visualiz_infer.permute(1, 2, 0).cpu().numpy()
            plt.imshow(visualiz_infer_np)
            plt.title('Image with Inferred Mask')
            plt.axis('off')

    plt.show()
    
def plot_losses(csv_path, metrics):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    # clean
    df = df.groupby('step').first().reset_index()
    df = df.astype('float16')
    df = df.astype({'epoch': 'int8', 'step': 'int8'})
    
    # Plot all metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3))
    
    # Plot each metric in its own subplot
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(df['step'], df[f'train_{metric}'], label=f'Training {metric}')
        ax.plot(df['step'], df[f'valid_{metric}'], label=f'Validation {metric}')
        ax.set_xlabel('Step')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Training and Validation {metric.capitalize()} Over Steps')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True)
