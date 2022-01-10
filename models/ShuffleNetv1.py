'''ShuffleNet in PyTorch.
See the paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()

        # 240 / 4 = 60
        mid_channels = int(out_channels / 4)

        # 第一个unit的输入通道数24太小，不进行分组卷积操作
        if in_channels == 24:
            self.groups = 1
        else:
            self.groups = groups

        # pointwise group convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # depthwise separable convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # another pointwise group convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))
        self.stride = stride

    def shuffle(self, x, groups):
        N, C, H, W = x.size()
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
        return out

    def forward(self, x):
        out = self.conv1(x)
        out = self.shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.conv3(out)

        # 如果是下采样，concat
        if self.stride == 2:
            res = self.shortcut(x)
            out = F.relu(torch.cat([out, res], 1))
        else:
            out = F.relu(out+x)
        return out



class ShuffleNetV1(nn.Module):
    def __init__(self, groups, channel_num, class_num=10):

        super().__init__()

        # 224x224x3 -> 112x112x24
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False)
        # 112x112x24 -> 56x56x24
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottleneck-11   240x28x28
        self.stage2 = self.make_layers(24, channel_num[0], 4, 2, groups)
        # Bottleneck-20   480x14x14
        self.stage3 = self.make_layers(channel_num[0], channel_num[1], 8, 2, groups)
        # Bottleneck-29   960x7x7
        self.stage4 = self.make_layers(channel_num[1], channel_num[2], 4, 2, groups)
        #  960x7x7 960x1x1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # -1x10
        self.fc = nn.Linear(channel_num[2], class_num)


    def make_layers(self, input_channels, output_channels, layers_num, stride, groups):
        layers = []
        layers.append(Bottleneck(input_channels, output_channels-input_channels, stride, groups))
        input_channels = output_channels
        for i in range(layers_num-1):
            Bottleneck(input_channels, output_channels, 1, groups)
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x



def ShuffleNetG2():
    return ShuffleNetV1(groups=2, channel_num=[200,400,800])


def ShuffleNetG3():
    return ShuffleNetV1(groups=3, channel_num=[240,480,960])






if __name__=="__main__":
    # model check
    model = ShuffleNetG3()
    summary(model, (3, 224, 224), device='cpu')
    from thop import profile

    input_size = (1, 3, 224, 224)
    flops, params = profile(model=model, input_size=input_size)
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))
    x = torch.randn(input_size)
    out = model(x)