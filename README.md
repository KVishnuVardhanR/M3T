# M3T- three-dimensional Medical image classifier using Multi-plane and Multi-slice Transformer

# Overview
To develop a novel deep learning method to classify Alzheimer’s disease

Alzheimer’s : A type of brain disorder that causes problems with memory, thinking and behaviour
![image](https://github.com/KVishnuVardhanR/M3T-/assets/33771427/c74b2d1d-bb67-4bc9-8310-a83b742af9f0)

# M3T Network Architecture:
![image](https://github.com/KVishnuVardhanR/M3T-/assets/33771427/60c4a501-86d3-41e7-9ddd-138bc4db6284)

# Let's start coding each part of the architecture
importing necessary libraries
>```
> import torch
> from torch import Tensor
> import torch.nn as nn
> import torchvision.models as models
> from einops import rearrange, reduce, repeat
> from einops.layers.torch import Rearrange, Reduce
> import torch.nn.functional as F

# 3D CNN block in M3T:
The authors mentioned that the shape of the output must match the shape of the input, the length, width and height of the output should be same as input

where I ∈ R^L×W×H , D3d : R^LxWxH -> R^C3xLxWxH

X = D3d(I), X ∈ R^C3d×L×W×H
![image](https://github.com/KVishnuVardhanR/M3T-/assets/33771427/1ad141da-6fd1-4aae-b669-0461fdab7a71)

>```
>class CNN3DBlock(nn.Module):
>    '''
>    To obtain 3D representation features, we apply 3D CNN block to the MRI image 
>    I ∈ R(L x W x H) where image length L, width W and height H are all the same.
>    X ∈ R(C3dxLxWxH)   where X = D3d(I)                                  Eq. (1)
>    Ref: 3.2. 3D Convolutional Neural Network Block 
>    '''
>    def __init__(self, in_channels, out_channels):
>        super(CNN3DBlock, self).__init__()
>
>        # 5 x 5 x 5 3D CNN
>        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, 
>                               stride=1, padding=2)
>        # Batch Normalization
>        self.bn1 = nn.BatchNorm3d(out_channels)
>
>        # ReLU
>        self.relu1 = nn.ReLU(inplace=True)
>
>        # 5 x 5 x 5 3D CNN
>        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=5, 
>                               stride=1, padding=2)
>       
>        # Batch Normalization
>        self.bn2 = nn.BatchNorm3d(out_channels)
>
>        # ReLU
>        self.relu2 = nn.ReLU(inplace=True)
>
>    def forward(self, x):
>        out = self.conv1(x)
>        out = self.bn1(out)
>        out = self.relu1(out)
>
>        out = self.conv2(out)
>        out = self.bn2(out)
>        out = self.relu2(out)
>
>        return out
