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
            predicted_mask = model.infer(image).squeeze(0)
        
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
            plt.figure(figsize=(5, 5))
            combined_overlay = utils.draw_segmentation_masks(image, masks=mask1 > 0.5, alpha=0.5, colors='green')
            if mask2 is not None:
                combined_overlay = utils.draw_segmentation_masks(combined_overlay, masks=mask2 > 0.5, alpha=0.5, colors='red')
            combined_overlay_np = combined_overlay.permute(1, 2, 0).cpu().numpy()
            plt.imshow(combined_overlay_np)
            plt.axis('off')

    plt.show()
    
def save_mask(image, mask1, path, mask2 = None):
    # convert
    image = F.convert_image_dtype(image, dtype=torch.uint8)
    # combine
    overlay = utils.draw_segmentation_masks(image, masks=mask1>0.5, alpha=0.6, colors='green')
    if mask2 is not None:
        overlay = utils.draw_segmentation_masks(overlay, masks=mask2 > 0.5, alpha=0.6, colors='red')
    
    combined_image = F.to_pil_image(overlay)
    combined_image.save(path)


def plot_losses(csv_path, metrics, test = False):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # drop epochs if we are plotting per image
    if test == True:
        df = df[df['epoch'].isnull()]
    
    # clean
    df = df.groupby('step').first().reset_index()
    
    # convert
    df = df.astype('float16')
    df = df.astype({'step': 'int16'})
    
    # Plot all metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3))
    
    # Plot each metric in its own subplot
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        if test == False:
            ax.plot(df['epoch'], df[f'train_{metric}'], label=f'Training {metric}')
            ax.plot(df['epoch'], df[f'valid_{metric}'], label=f'Validation {metric}')
            ax.set_xlabel('Epoch')
        else:
            ax.scatter(df['step'], df[f'test_{metric}'], label=f'Test {metric}', s=5)
            ax.set_xlabel('Image index')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True)
