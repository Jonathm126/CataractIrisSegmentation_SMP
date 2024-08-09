# Cataract_Iris_Segmentation_SMP
 
# configuration
The config .json has the following information:

{
    "NAME": "Unetpp_mobilenetv3_large_100_pretrained", # name of the model \ experiment
    "MODE": "train", # "train" or "load"
    "NUM_EPOCHS": 10, # epochs to train

    "CLASSES": ["pupil"], # not relevant
    "IN_CH": 3, # rgb

    "ARCH": "UnetPlusPlus",
    "ENCODER": "timm-mobilenetv3_large_100",
    "ENCODER_WEIGHTS": "imagenet",
    "LOSS": "DiceLoss",

    "LR": 0.00005
}
