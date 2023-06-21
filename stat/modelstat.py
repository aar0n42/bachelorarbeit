import torch
import torch.nn as nn
#from torchsummary import summary
from torchinfo import summary

import segmentation_models_pytorch as smp
from Models import fcbformer

model = fcbformer.FCBFormer()

model.to('cuda')
input_size = (1,3, 640, 640)  # Example: (C, H, W) for an RGB image of size 224x224

# Print the summary
summary(model, input_size)
