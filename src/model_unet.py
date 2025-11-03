# forest_fire_detection/src/model_unet.py

import torch
from segmentation_models_pytorch import Unet

def get_unet(in_channels=3, classes=1):
    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes
    )
    return model
