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


# Extraction of Multi-plane, Multi slice images and  2D CNN block in M3T:
After using 3D CNN block into the input image, the multi-plane and multi-slice image features is extracted from the 3D representation features X. The features are calculated from the extraction operator E. The
operator consists of coronal features extractor Ecor : R^C3d×L×W×H → R^C3d×N×W×H.

However, the authors did not mention what method they have used to achieve the above. This can be achieved by using splitting across the dimension and concatenation:
>```
>class MultiPlane_MultiSlice_Extract_Project(nn.Module):
>    '''
>    The multi-plane and multi-slice image features extraction from the 3D 
>    representation features X and applying 2D CNN followed by Non-Linear
>    Projection
>    Ref: 3.3. Extraction of Multi-plane, Multi slice images and 
>         3.4. 2D Convolutional Neural Network Block
>    '''
>    def __init__(self, out_channels: int):
>        super(MultiPlane_MultiSlice_Extract_Project, self).__init__()
>
>    def forward(self, input_tensor):
>        # Extract coronal features
>        coronal_slices = torch.split(input_tensor, 1, dim=2) # This gives us a tuple of length 128, where each element has shape (batch_size, channels, 1, length, width) 
>        Ecor = torch.cat(coronal_slices, dim=2) # lets concatenate along dimension 2 to get the desired output shape for Ecor: R^C3d×N×W×H.
>        return Ecor


We can get the axial ans saggital features Esag, Eax using the above method. 

Now after extracting [Ecor, Esag and Eax] = E, **The authors calculated multi-plane multi-slice features S = [Scor, Ssag, Sax] using E = [Ecor, Esag, Eax] from 3D representation features X**. We can use basic matrix multiplication to achieve it. We will also complete the 2D CNN block part within this class:

>```
>class MultiPlane_MultiSlice_Extract_Project(nn.Module):
>    '''
>    The multi-plane and multi-slice image features extraction from the 3D 
>    representation features X and applying 2D CNN followed by Non-Linear
>    Projection
>    N = length = width = height based on the mentioned input size in the paper
>    Ref: 3.3. Extraction of Multi-plane, Multi slice images and 
>         3.4. 2D Convolutional Neural Network Block
>    '''
>    def __init__(self, out_channels: int):
>        super(MultiPlane_MultiSlice_Extract_Project, self).__init__()
>        # 2D CNN part
>        # Load the pre-trained ResNet-18 model and Extract the global average pooling layer
>        self.gap_layer = models.resnet50(pretrained=True).avgpool
>
>        # Non - Linear Projection block
>        self.non_linear_proj = nn.Sequential(
>            nn.Linear(out_channels, 512),
>            nn.ReLU(),
>            nn.Linear(512, 256)
>            )
>
>    def forward(self, input_tensor):
>        # Extract coronal features
>        coronal_slices = torch.split(input_tensor, 1, dim=2)                      # This gives us a tuple of length 128, where each element has shape (batch_size, channels, 1, width, height) 
>        Ecor = torch.cat(coronal_slices, dim=2)                                   # lets concatenate along dimension 2 to get the desired output shape for Ecor: R^C3d×N×W×H.
>
>        saggital_slices = torch.split(input_tensor.clone(), 1, dim = 3)           # This gives us a tuple of length 128, where each element has shape (batch_size, channels, length, 1, height) 
>        Esag = torch.cat(saggital_slices, dim = 3)                                # lets concatenate along dimension 3 to get the desired output shape for Ecor: R^C3d×L×N×H.
>
>        axial_slices = torch.split(input_tensor.clone(), 1, dim = 4)              # This gives us a tuple of length 128, where each element has shape (batch_size, channels, length, width, 1) 
>        Eax = torch.cat(axial_slices, dim = 4)                                    # lets concatenate along dimension 3 to get the desired output shape for Ecor: R^C3d×L×W×N.
>
>        # Lets calculate S using E for X
>        # after matirx multiplications, we reshape the outputs based on its plane for concatenation 
>        Scor = (Ecor * input_tensor).permute(0, 2, 1, 3, 4).contiguous()          # Scor will now have a shape (batch_size, N, channels, width, height) 
>        Ssag = (Esag * input_tensor).permute(0, 3, 1, 2, 4).contiguous()          # Ssag will now have a shape (batch_size, N, channels, length, height)
>        Sax  =  (Eax * input_tensor).permute(0, 4, 1, 2, 3).contiguous()          # Sax will now have a shape  (batch_size, N, channels, length, width)
>        
>        # Concatenate the reshaped extracted features : R(C3d×L×W×H) → R(3N×C3d×L×L)
>        S = torch.cat((Scor, Ssag, Sax), dim = 1)                                 # Now S will have a shape of (batch_size, 3N, channels, length, length)
>
>        # 2D CNN block
>        # perform global average pooling using pre-trained ResNet50 network
>        # D2d : R(3N×C3d×L×L) → R(3N×C2d)    (C2d is out channel size of 2D CNN)
>        pooled_feat = self.gap_layer(S)                                           #  Eq. (4)
>
>        # Non-Linear Projection part T ∈ R(3N×d)     (d is projection dimension)
>        output_tensor = self.non_linear_proj(pooled_feat.squeeze(dim=3).squeeze(dim=3)) # remove the extra dimensions to get the desired output shape
>        return output_tensor
