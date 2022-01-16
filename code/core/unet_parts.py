""" Parts of the U-Net model """
import math

from torchvision.transforms import ToPILImage, transforms

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv(nn.Module):
    """(convolution => [BN] => ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_Cat(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
        )
        self.doubleconv = DoubleConv(in_channels+out_channels, out_channels)
    def forward(self, x1,x2):
        x1=self.maxpool_conv(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.doubleconv(x)

class Down(nn.Module):
    """Downscaling with DualAttention then maxpool """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ca=ChannelAttention(out_channels)
        self.sa=SpatialAttention()
        self.conv=Conv(in_channels,out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # DoubleConv(in_channels, out_channels)
        )
    def forward(self, x1):
        x = self.maxpool_conv(x1)
        x0 = self.conv(x)
        x = self.ca(x0) * x0
        x = self.sa(x) * x

        # if self.downsample is not None:
        #     residual = self.downsample(x1)

        x += x0
        return self.relu(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)
        # self.contra=nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        # 这个上采用需要设置其输入通道，输出通道.其中kernel_size、stride
        # 大小要跟对应下采样设置的值一样大小。这样才可恢复到相同的wh。这里时反卷积操作。
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        # 这里不会改变通道数，其中scale_factor是上采用的放大因子，其是相对于当前的输入大小的倍数
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=(in_channels//32), align_corners=True))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Sequential(nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1),
        nn.BatchNorm2d(out_channels))
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.Sequential(
            upconv2x2(in_channels, out_channels,mode=""),
            conv1x1(in_channels, out_channels),
        )
        # nn.Upsample(scale_factor=in_channels//32, mode='nearest'),
        # nn.Conv2d(out_channels,out_channels, kernel_size=1)

    def forward(self, x):
        x=self.up(x)
        outlayer = nn.Sigmoid()
        return outlayer(x)


class OutConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels, kernel_size=1),
        )

    def forward(self, x):
        x=self.conv(x)
        outlayer = nn.Sigmoid()
        return outlayer(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)   #config.hidden_size768
        for _ in range(12):   #num_layers" 12
            layer = Block(hidden_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)



class Embeddings(nn.Module):

    def __init__(self, hidden_size, img_size, in_channels):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        patch_size = (img_size[0] // 16 , img_size[1] // 16 )
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768)) #hidden_size768

        self.dropout = nn.Dropout(0.1)

        self.encoder = Encoder(hidden_size)

        self.conv_more = Conv2dReLU(
            hidden_size//n_patches,
            in_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)    #  torch.Size([1, 768, 16, 16])torch.Size([1, 16, 768])torch.Size([1, 256, 768])
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        encoded = self.encoder(embeddings)

        B, n_patch, hidden = encoded.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(n_patch), int(n_patch)
        x = encoded.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden//n_patch, h, w)
        x = self.conv_more(x)
        return x
class Block(nn.Module):
    def __init__(self, hidden_size):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(hidden_size)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x
class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.num_attention_heads = 12
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(0)
        self.proj_dropout = nn.Dropout(0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


