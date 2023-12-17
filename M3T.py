import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from torchsummary import summary

class CNN3DBlock(nn.Module):
    '''
    To obtain 3D representation features, we apply 3D CNN block to the MRI image 
    I ∈ R(L x W x H) where image length L, width W and height H are all the same.
    X ∈ R(C3dxLxWxH)   where X = D3d(I)                                  Eq. (1)
    Ref: 3.2. 3D Convolutional Neural Network Block 
    '''
    def __init__(self, in_channels, out_channels):
        super(CNN3DBlock, self).__init__()

        # 5 x 5 x 5 3D CNN
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, 
                              stride=1, padding=2)
        # Batch Normalization
        self.bn1 = nn.BatchNorm3d(out_channels)

        # ReLU
        self.relu1 = nn.ReLU(inplace=True)

        # 5 x 5 x 5 3D CNN
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=5, 
                              stride=1, padding=2)
      
        # Batch Normalization
        self.bn2 = nn.BatchNorm3d(out_channels)

        # ReLU
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out

class MultiPlane_MultiSlice_Extract_Project(nn.Module):
    '''
    The multi-plane and multi-slice image features extraction from the 3D 
    representation features X and applying 2D CNN followed by Non-Linear
    Projection
    N = length = width = height based on the mentioned input size in the paper
    Ref: 3.3. Extraction of Multi-plane, Multi slice images and 
         3.4. 2D Convolutional Neural Network Block
    '''
    def __init__(self, out_channels: int):
        super(MultiPlane_MultiSlice_Extract_Project, self).__init__()
        # 2D CNN part
        # Load the pre-trained ResNet-18 model and Extract the global average pooling layer
        self.gap_layer = models.resnet50(pretrained=True).avgpool

        # Non - Linear Projection block
        self.non_linear_proj = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
            )

    def forward(self, input_tensor):
        # Extract coronal features
        coronal_slices = torch.split(input_tensor, 1, dim=2)                      # This gives us a tuple of length 128, where each element has shape (batch_size, channels, 1, width, height) 
        Ecor = torch.cat(coronal_slices, dim=2)                                   # lets concatenate along dimension 2 to get the desired output shape for Ecor: R^C3d×N×W×H.

        saggital_slices = torch.split(input_tensor.clone(), 1, dim = 3)           # This gives us a tuple of length 128, where each element has shape (batch_size, channels, length, 1, height) 
        Esag = torch.cat(saggital_slices, dim = 3)                                # lets concatenate along dimension 3 to get the desired output shape for Ecor: R^C3d×L×N×H.

        axial_slices = torch.split(input_tensor.clone(), 1, dim = 4)              # This gives us a tuple of length 128, where each element has shape (batch_size, channels, length, width, 1) 
        Eax = torch.cat(axial_slices, dim = 4)                                    # lets concatenate along dimension 3 to get the desired output shape for Ecor: R^C3d×L×W×N.

        # Lets calculate S using E for X
        # after matirx multiplications, we reshape the outputs based on its plane for concatenation 
        Scor = (Ecor * input_tensor).permute(0, 2, 1, 3, 4).contiguous()          # Scor will now have a shape (batch_size, N, channels, width, height) 
        Ssag = (Esag * input_tensor).permute(0, 3, 1, 2, 4).contiguous()          # Ssag will now have a shape (batch_size, N, channels, length, height)
        Sax  =  (Eax * input_tensor).permute(0, 4, 1, 2, 3).contiguous()          # Sax will now have a shape  (batch_size, N, channels, length, width)
       
        # Concatenate the reshaped extracted features : R(C3d×L×W×H) → R(3N×C3d×L×L)
        S = torch.cat((Scor, Ssag, Sax), dim = 1)                                 # Now S will have a shape of (batch_size, 3N, channels, length, length)

        # 2D CNN block
        # perform global average pooling using pre-trained ResNet50 network
        # D2d : R(3N×C3d×L×L) → R(3N×C2d)    (C2d is out channel size of 2D CNN)
        pooled_feat = self.gap_layer(S).squeeze(dim=3).squeeze(dim=3)             #  Eq. (4)

        # Non-Linear Projection part T ∈ R(3N×d)     (d is projection dimension)
        output_tensor = self.non_linear_proj(pooled_feat)                         # Now we have the desired output shape
        return output_tensor


class EmbeddingLayer(nn.Module):
    '''
    After calculating the multi-plane and multi-slice image tokens, position and 
    plane embedding tokens are added to the image tokens from non-linear projection layer.
    Ref. 3.5. Position and Plane Embedding Block
    emb_size = d = 256, total_tokens = 3S = 3*128 = 384
    where d = attention dimension and S = input size
    '''
    def __init__(self, emb_size: int = 256, total_tokens: int = 384):
        super(EmbeddingLayer, self).__init__()

        # zcls ∈ R(d)
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))

        # zsep ∈ R(d)
        self.sep_token = nn.Parameter(torch.randn(1,1, emb_size))

        # Ppos ∈ R((3S+4)×d)
        self.plane = nn.Parameter(torch.randn(total_tokens + 4, emb_size))

        # Ppos ∈ R((3S+4)×d)
        self.positions = nn.Parameter(torch.randn(total_tokens + 4, emb_size))

    def forward(self, input_tensor):
        b, _, _ = input_tensor.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        sep_token = repeat(self.sep_token, '() n e -> b n e', b=b)

        x = torch.cat((cls_tokens, input_tensor[:, :128, :], sep_token, input_tensor[:, 128:256, :], sep_token, input_tensor[:, 256:, :], sep_token), dim=1)

        x += self.plane

        x += self.positions
       
        # the above represents Eq. (6)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 256, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 3, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )



class TransformerEncoderBlock(nn.Sequential):
    '''
    We keep the forward expansion as 3, since all the hidden and MLP sizes must be 768 in the transformer encoder
    Ref. 3.6. Transformer Block
    '''
    def __init__(self,
                 emb_size: int = 256,
                 drop_p: float = 0.,
                 forward_expansion: int = 3,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            # Zk = MSA(LN(Zk)) + Zk                                      Eq. (7)
            ResidualAdd(nn.Sequential(
                # Layer Normalization (LN)
                nn.LayerNorm(emb_size),

                # Multi Head Self attention (MSA)
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            # Zk+1 = MLP(LN(Zk))+ Zk                                     Eq. (8)
            ResidualAdd(nn.Sequential(
                # MLP blocks
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    '''
    The number 256 is same with projection dimension (attention dimension) d used in the transformer.
    The number of transformer layers is 8. The hidden size and MLP size are 768, 
    and the number of heads = 8.
    '''
    def __init__(self, depth: int = 8, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Module):
    '''
    A linear classifier is used to classify the encoded input based on the MLP 
    head: ZKcls ∈ R(d). There are two final categorization classes: NC and AD.
    The first token (cls_token) from the sequence is used for classification.
    '''
    def __init__(self, emb_size: int = 256, n_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        # As x is of shape [batch_size, num_tokens, emb_size]
        # and the cls_token is the first token in the sequence
        cls_token = x[:, 0, :]
        cls_token = self.norm(cls_token)
        return self.linear(cls_token)
        
class M3T(nn.Sequential):
    def __init__(self,
                in_channels: int = 1,
                out_channels: int = 32,
                emb_size: int = 256,
                depth: int = 8,
                n_classes: int = 2,
                **kwargs):
        super().__init__(
            CNN3DBlock(in_channels, out_channels),
            MultiPlane_MultiSlice_Extract_Project(out_channels),
            EmbeddingLayer(emb_size=emb_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

# from torchsummary import summary
# model = M3T()
# summary(model, (1, 128, 128, 128))
