import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary


# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.

def StemConv(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    """
    Performing 3x3 convolution in the first layer
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            ReLU(inplace=True)
        )



def PointwiseConv(inp, oup, use_batch_norm=True, onnx_compatible=False):
    """
    Performing 1x1 pointwise convolution
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            ReLU(inplace=True)
        )



class ChannelShuffleBlock(nn.Module):
    """
    Channel shuffle operation from ShuffleNet
    """
    def __init__(self, groups=3):
        super(ChannelShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)



class EqualSplitBlock(nn.Module):
    """
    It is responsible for distributing the same subsets of feature maps to the 3 parallel convolutions
    """
    def __init__(self, n):
        super(EqualSplitBlock, self).__init__()
        self.n = n

    def forward(self, x):
        c3 = int(x.size(1))
        c1 = int(x.size(1) // self.n)
        c2 = c3 - c1
        return x[:, :c1, :, :], x[:, c1:c2, :, :], x[:, c2:c3, :, :]



class DimensionTranspose(nn.Module):
    """
    tensor dimension transpose
    """
    def __init__(self, spatial_type="heightwise"):
        super(DimensionTranspose, self).__init__()
        self.spatial_type = spatial_type

    def forward(self, x):
        #N, C, H, W = x.size()
        if self.spatial_type == "heightwise":
            # 交换 C, H两个维度
            x = x.transpose(1,2)
        elif self.spatial_type == "widthwise":
            # 交换 C, W两个维度
            x = x.transpose(1,3)
        return x



def PointWiseGroupConv(channel_in, channel_out, group_number=1, stride=1, use_batch_norm=True, onnx_compatible=False):
    """
    Pointwise 1x1 Group Convolution

    group number is 3 for general-wise convolution (depthwise, heightwise, and widthwise convolutions)

    its task is to project the input feature map into expansion layer for feature abstraction
    it is also responsible for subsampling.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return (
            nn.Sequential(
                nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=stride, padding=0, groups=group_number, bias=False),
                nn.BatchNorm2d(channel_out),
                ReLU(inplace=True)
            )
        )
    else:
        return (
            nn.Sequential(
                nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=stride, padding=0, groups=group_number, bias=False),
                ReLU(inplace=True)
            )
        )



def LinearConv(channel_in, channel_out, use_batch_norm=True):
    """
    The 1x1 linear convolution with ReLU
    """
    if use_batch_norm:
        return (
            nn.Sequential(
                nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(channel_out)
            )
        )
    else:
        return (
            nn.Sequential(
                nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            )
        )




def DepthWiseConv(in_channels, kernel_size=3, stride=1, use_batch_norm=True, onnx_compatible=False):
    """
    Depthwise Convolution
    in_channels == out_channels == groups

    its task is responsible for feature abstraction
    can also perform subsampling
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=1, groups=in_channels, bias=False),
            ReLU(inplace=True)
        )



def SpatialwiseConv(channel_proxy, kernel_size=3, stride=1, use_batch_norm=True, onnx_compatible=False):
    """
    Similar to Depthwise Conv, but on different domains
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return (
            nn.Sequential(
                nn.Conv2d(in_channels=channel_proxy, out_channels=channel_proxy, kernel_size=kernel_size, stride=stride, padding=1, groups=channel_proxy, bias=False),
                nn.BatchNorm2d(channel_proxy),
                ReLU(inplace=True)
            )
        )
    else:
        return (nn.Sequential(
                nn.Conv2d(in_channels=channel_proxy, out_channels=channel_proxy, kernel_size=kernel_size, stride=stride, padding=1, groups=channel_proxy, bias=False),
                ReLU(inplace=True)
            )
        )



class GeneralWiseSeparableConv(nn.Module):
    def __init__(self, in_channel, hidden_dim, out_channel, input_height, input_width, stride=1, use_batch_norm=True, onnx_compatible=False):
        super(GeneralWiseSeparableConv, self).__init__()

        # if stride ==2, the width and height of feature maps are reduced by 1/2
        if stride == 2:
            input_height = input_height // 2
            input_width = input_width // 2
        else:
            input_height = input_height
            input_width = input_width

        assert hidden_dim % 3 == 0
        subset_in_dim = hidden_dim // 3

        self.pointwiseGroupConv = PointWiseGroupConv(channel_in=in_channel, channel_out=hidden_dim, group_number=3, stride=stride,
                                                     use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible)
        # channel shuffling
        self.channelShuffle = ChannelShuffleBlock(groups=3)

        # channel splitting
        self.split = EqualSplitBlock(n=3)

        # depthwise convolution in the first branch of parallels
        self.depthwiseConv = DepthWiseConv(in_channels=subset_in_dim, kernel_size=3, stride=1, use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible)

        # dimension transpose for heightwise convolution  [-1, 48, 112, 112] -> [-1, 112, 48, 112]
        self.heightwiseTranspose = DimensionTranspose(spatial_type="heightwise")

        # heightwise convolution in the second branch of parallels [-1, 112, 48, 112]->[-1, 112, 48, 112]
        self.heightwiseConv = SpatialwiseConv(channel_proxy=input_height, kernel_size=3, stride=1, use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible)

        # dimension transpose for heightwise convolution  [-1, 48, 112, 112] -> [-1, 112, 112, 48]
        self.widthwiseTranspose = DimensionTranspose(spatial_type="widthwise")

        # widthwise convolution in the third branch of parallels [-1, 112, 112, 48] -> [-1, 112, 112, 48]
        self.widthwiseConv = SpatialwiseConv(channel_proxy=input_width, kernel_size=3, stride=1, use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible)

        # the final linear projection convolution
        self.linearConv = LinearConv(channel_in=hidden_dim, channel_out=out_channel, use_batch_norm=use_batch_norm)

    def forward(self, x):
        expansion_layer = self.pointwiseGroupConv(x)
        expansion_shuffle = self.channelShuffle(expansion_layer)
        split1, split2, split3 = self.split(expansion_shuffle)
        split1 = self.depthwiseConv(split1)
        split2 = self.widthwiseTranspose(split2)
        split2 = self.widthwiseConv(split2)
        split2 = self.widthwiseTranspose(split2)
        split3 = self.widthwiseTranspose(split3)
        split3 = self.widthwiseConv(split3)
        split3 = self.widthwiseTranspose(split3)
        # residual connection to improve gradient flow and feature reuse
        concat_tensor = torch.cat([split1, split2, split3], 1) + expansion_shuffle
        # linear projection to output feature map
        out = self.linearConv(concat_tensor)
        return out



def make_divisible(x, divisible_by=3):
    """ function used for width multiplier """
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)



class GSCBlock(nn.Module):

    def __init__(self, input_height, input_width, channel_in, channel_out, expand_ratio, stride, use_batch_norm=True, onnx_compatible=False):
        super(GSCBlock, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        # the dimension of expansion layer: 24 * 6  = 144
        hidden_dim = round(channel_in * expand_ratio)
        self.use_res_connect = self.stride == 1 and channel_in == channel_out

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                GeneralWiseSeparableConv(in_channel=hidden_dim, hidden_dim=hidden_dim, out_channel=channel_out,
                                         input_height=input_height, input_width=input_width, stride=stride,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible)
            )
        else:
            self.conv = nn.Sequential(
                GeneralWiseSeparableConv(in_channel=channel_in, hidden_dim=hidden_dim, out_channel=channel_out,
                                         input_height=input_height, input_width=input_width, stride=stride,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class GSCNetOneDepthTwoWidth(nn.Module):
    def __init__(self, input_height=224, input_width=224, in_channels=3, n_classes=1000, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(GSCNetOneDepthTwoWidth, self).__init__()

        in_planes = 36
        last_channel = 1280

        gsc_block_configs = [
            # t, c, n, s
            [1, 18, 1, 1],  # 112 * 112 * 36 -> 112 * 112 * 18
            [6, 24, 2, 2],  # 112 * 112 * 18 -> 56 * 56 * 24
            [6, 36, 3, 2],  # 56 * 56 * 24 -> 28 * 28 * 36
            [6, 60, 4, 2],  # 28 * 28 * 36 -> 14 * 14 * 60
            [6, 96, 3, 1],  # 14 * 14 * 60 -> 14 * 14 * 96
            [6, 162, 3, 2],  # 14 * 14 * 96 -> 7 * 7 * 162
            [6, 321, 1, 1]  # 7 * 7 * 162 -> 7 * 7 * 321
        ]

        # width multiplier
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # 因为第一次下采样，特征图谱的长和宽变为input的1/2
        self.input_height = input_height // 2
        self.input_width = input_width // 2

        # 第一次卷积, 并下采样 224x224x3 -> 112x112x36
        assert input_height % 32 == 0
        assert input_width % 32 == 0
        in_planes = make_divisible(in_planes * width_mult) if width_mult > 1.0 else in_planes

        # build the stem convolution
        self.features = [StemConv(inp=in_channels, oup=in_planes, stride=2, use_batch_norm=True, onnx_compatible=False)]

        # t= expansion factor, c = output channel number, n = bottleneck number, s = stride
        for t, c, n, s in gsc_block_configs:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                # bottleneck number > 1, 总是第一个下采样
                stride = s if i == 0 else 1
                self.features.append(GSCBlock(input_height=self.input_height, input_width=self.input_width,
                                       channel_in=in_planes, channel_out=output_channel, expand_ratio=t,
                                       stride=stride, use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))

                in_planes = output_channel

                if stride == 2:
                    self.input_height = self.input_height // 2
                    self.input_width = self.input_width // 2
                else:
                    self.input_height = self.input_height
                    self.input_width = self.input_width

        # building last several layers
        self.features.append(PointwiseConv(inp=in_planes, oup=self.last_channel, use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # 分类器
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(self.last_channel, n_classes),
        )

        # 初始化权重
        self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



if __name__=="__main__":
    model = GSCNetOneDepthTwoWidth(input_height=320, input_width=320,
                                   in_channels=3, n_classes=1000, width_mult=1., dropout_ratio=0.2,
                                   use_batch_norm=True, onnx_compatible=False)

    summary(model, (3, 320, 320), device='cpu')
    from thop import profile

    input_size = (1, 3, 320, 320)
    flops, params = profile(model=model, input_size=input_size)
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))





