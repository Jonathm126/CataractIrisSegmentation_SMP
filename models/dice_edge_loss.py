import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def extract_volume_2D_contours_pt(mask, input_shape, th=1):
    kernel = torch.zeros((3, 3, input_shape[2]), dtype=torch.int32)
    if th == 0.5:
        int_mask = torch.round(mask).int()
    elif th == 1:
        int_mask = mask.int()
    eroded_mask = F.conv2d(int_mask, kernel, stride=1, padding=1)
    contour = torch.bitwise_xor(int_mask, eroded_mask).float()
    return contour

def contour_dice_pt(y_true, y_pred, input_shape, smooth=1., th=1):
    true_contours = extract_volume_2D_contours_pt(y_true[:, 0, :, :, :], input_shape, th)
    pred_contours = extract_volume_2D_contours_pt(y_pred[:, 0, :, :, :], input_shape, th)
    return dice_coefficient_pt(true_contours, pred_contours, smooth)

def contour_dice_loss(y_true, y_pred, input_shape, gamma, smooth=1.):
    # dice function
    dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
    
    
    # compute loss
    dice_loss = dice(y_true, y_pred, smooth)
    
    contour_dice_loss = contour_dice_pt(y_true, y_pred, input_shape, smooth)
    return -dice - (gamma * contour_dice_loss)
