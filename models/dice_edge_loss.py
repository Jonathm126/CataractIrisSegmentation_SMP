import torch
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, BINARY_MODE

def extract_volume_2D_contours_pt(mask, N, th=1):
    kernel = torch.ones((N ,1, 3, 3), dtype=torch.int32) # N, C, H, W
    if th == 0.5:
        int_mask = torch.round(mask).int()
    elif th == 1:
        int_mask = mask.int()
    int_mask = int_mask.cpu()
    eroded_mask = F.conv2d(int_mask, weight=kernel, stride=1, padding=1)
    contour = torch.bitwise_xor(int_mask, eroded_mask).float()
    return contour


def contour_dice_loss(y_true, y_pred, gamma=1, th = 1):
    # dice function
    dice = DiceLoss(BINARY_MODE)
    
    # compute loss
    dice_loss = dice(y_true, y_pred)
    
    # compute contour loss
    N = y_true.shape[0]
    true_contours = extract_volume_2D_contours_pt(y_true[:, :, :, :], N, th)
    pred_contours = extract_volume_2D_contours_pt(y_pred[:, :, :, :], N, th)
    contour_dice_loss = dice(true_contours, pred_contours)
    
    return dice_loss + (gamma * contour_dice_loss)
