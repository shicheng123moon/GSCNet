import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math



def DepthWiseConv(in_channels, stride=1):
    """
    Depthwise Separable Convolution
    in_channels == out_channels == groups

    its task is responsible for feature abstraction
    can also perform subsampling
    """
    return (
        nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
    )



def PointWiseGroupConv(channel_in, channel_out, group_number=1, stride=1):
    """
    Pointwise 1x1 Group Convolution

    group number is 3 for general-wise convolution (depthwise, heightwise, and widthwise convolutions)

    its task is to project the input feature map into expansion layer for feature abstraction
    it is also responsible for subsampling.

    """
    return (
        nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=stride, padding=0, groups=group_number, bias=False),
            nn.BatchNorm2d(channel_out),
            nn.ReLU6(inplace=True)
        )
    )



def PointWiseConv(channel_in, channel_out):
    """
    The regular 1x1 convolution without subsmpling
    """
    return (
        nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=1,  padding=0, groups=1, bias=False),
            nn.BatchNorm2d(channel_out),
            nn.ReLU6(inplace=True)
        )
    )



def LinearConv(channel_in, channel_out):
    """
    The 1x1 linear convolution with ReLU
    """
    return (
        nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(channel_out)
        )
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



class SpatialwiseConv(nn.Module):
    """
    Spatial-wise Convolution: heightwise convolution and widthwise convolution
    now the grouping is performed on width dimension and height dimension
    """
    def __init__(self, channel_proxy, stride=1):
        super(SpatialwiseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel_proxy, out_channels=channel_proxy, kernel_size=3,
                              stride=stride, padding=1, groups=channel_proxy, bias=False)
        self.bn = nn.BatchNorm2d(channel_proxy)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



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




class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwiseConv = DepthWiseConv(in_channels=in_channel, stride=stride)
        self.linearConv = LinearConv(channel_in=in_channel, channel_out=out_channel)

    def forward(self, x):
        x = self.depthwiseConv(x)
        x = self.linearConv(x)
        return x




class SpatialWiseSeparableConv(nn.Module):
    def __init__(self, in_channel, hidden_dim, out_channel, input_height, input_width, stride=1):
        super(SpatialWiseSeparableConv, self).__init__()

        # if stride ==2, the width and height of feature maps are reduced by 1/2
        if stride == 2:
            input_height = input_height // 2
            input_width = input_width // 2
        else:
            input_height = input_height
            input_width = input_width

        assert hidden_dim % 3 == 0
        subset_in_dim = hidden_dim // 3

        self.pointwiseGroupConv = PointWiseGroupConv(channel_in=in_channel, channel_out=hidden_dim, group_number=3, stride=stride)

        # channel shuffling
        self.channelShuffle = ChannelShuffleBlock(groups=3)

        # channel splitting
        self.split = EqualSplitBlock(n=3)

        # depthwise convolution in the first branch of parallels
        self.depthwiseConv = DepthWiseConv(in_channels=subset_in_dim, stride=1)

        # dimension transpose for heightwise convolution  [-1, 48, 112, 112] -> [-1, 112, 48, 112]
        self.heightwiseTranspose = DimensionTranspose(spatial_type="heightwise")

        # heightwise convolution in the second branch of parallels [-1, 112, 48, 112]->[-1, 112, 48, 112]
        self.heightwiseConv = SpatialwiseConv(channel_proxy=input_height, stride=1)

        # dimension transpose for heightwise convolution  [-1, 48, 112, 112] -> [-1, 112, 112, 48]
        #self.widthwiseTranspose = DimensionTranspose(spatial_type="widthwise")

        # widthwise convolution in the third branch of parallels [-1, 112, 112, 48] -> [-1, 112, 112, 48]
        #self.widthwiseConv = SpatialwiseConv(channel_proxy=input_width, stride=1)

        # the final linear projection convolution
        self.linearConv = LinearConv(channel_in=hidden_dim, channel_out=out_channel)

    def forward(self, x):
        expansion_layer = self.pointwiseGroupConv(x)
        expansion_shuffle = self.channelShuffle(expansion_layer)
        split1, split2, split3 = self.split(expansion_shuffle)
        split1 = self.heightwiseTranspose(split1)
        split1 = self.heightwiseConv(split1)
        split1 = self.heightwiseTranspose(split1)
        split2 = self.heightwiseTranspose(split2)
        split2 = self.heightwiseConv(split2)
        split2 = self.heightwiseTranspose(split2)
        split3 = self.heightwiseTranspose(split3)
        split3 = self.heightwiseConv(split3)
        split3 = self.heightwiseTranspose(split3)
        # residual connection to improve gradient flow and feature reuse
        concat_tensor = torch.cat([split1, split2, split3], 1) + expansion_shuffle
        # linear projection to output feature map
        out = self.linearConv(concat_tensor)
        return out




class GSCBlock(nn.Module):

    def __init__(self, input_height, input_width, channel_in, channel_out, expand_ratio, stride):
        super(GSCBlock, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        # use the residual shortcut when stride == 1 and channel_in equals channel_out
        self.use_res_connect = self.stride == 1 and channel_in == channel_out

        # the dimension of expansion layer: 24 * 6  = 144
        hidden_dim = channel_in * expand_ratio

        if expand_ratio == 1:
            self.conv = nn.Sequential(DepthwiseSeparableConv(in_channel=hidden_dim, out_channel=channel_out, stride=stride))
        else:
            self.conv = nn.Sequential(SpatialWiseSeparableConv(in_channel=channel_in, hidden_dim=hidden_dim, out_channel=channel_out,
                                                               input_height=input_height, input_width=input_width, stride=stride))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class GSCNetHeightOnly(nn.Module):
    def __init__(self, input_height=224, input_width=224, in_channels=3, n_classes=1000):
        super(GSCNetHeightOnly, self).__init__()

        self.in_planes = 36

        # 因为第一次下采样，特征图谱的长和宽变为input的1/2
        self.input_height = input_height // 2
        self.input_width = input_width // 2

        # 第一次卷积，并下采样 224x224x3 -> 112x112x36
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=36, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(36)

        layers = []

        self.gsc_block_configs = [
            # t, c, n, s
            [1, 18, 1, 1],   # 112 * 112 * 36 -> 112 * 112 * 18
            [6, 24, 2, 2],   # 112 * 112 * 18 -> 56 * 56 * 24
            [6, 36, 3, 2],   # 56 * 56 * 24 -> 28 * 28 * 36
            [6, 60, 4, 2],   # 28 * 28 * 36 -> 14 * 14 * 60
            [6, 96, 3, 1],   # 14 * 14 * 60 -> 14 * 14 * 96
            [6, 162, 3, 2],  # 14 * 14 * 96 -> 7 * 7 * 162
            [6, 321, 1, 1]   # 7 * 7 * 162 -> 7 * 7 * 321
        ]

        # t= expansion factor, c = output channel number, n = bottleneck number, s = stride
        for t, c, n, s in self.gsc_block_configs:
            for i in range(n):
                # bottleneck number > 1, 总是第一个下采样
                stride = s if i == 0 else 1
                layers.append(GSCBlock(input_height=self.input_height, input_width=self.input_width,
                                       channel_in=self.in_planes, channel_out=c, expand_ratio=t, stride=stride))
                self.in_planes = c

                if stride == 2:
                    self.input_height = self.input_height // 2
                    self.input_width = self.input_width // 2
                else:
                    self.input_height = self.input_height
                    self.input_width = self.input_width

        self.layers = nn.Sequential(*layers)

        # 最后一次卷积
        self.last_conv = PointWiseConv(self.in_planes, 1280)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 分类器
        self.classifier = nn.Linear(1280, n_classes)

        # 初始化权重
        self._initialize_weights()


    def forward(self, x):
        x = F.relu(self.bn1(self.stem_conv(x)))
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
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
    model = GSCNetHeightOnly(n_classes=1000)
    summary(model, (3, 224, 224), device='cpu')
    from thop import profile

    input_size = (1, 3, 224, 224)
    flops, params = profile(model=model, input_size=input_size)
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))
