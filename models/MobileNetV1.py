import torch.nn as nn
from torchsummary import summary
import torch

class MobileNetV1(nn.Module):

    """ MobileNetV1 has a lot of expensive 1*1 convolution
    https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV1.py
    """

    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # depthwise separable convolution
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pointwise convolution
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2), # 224 * 224 * 3 -> 112 * 112 * 32
            conv_dw(32, 64, 1),    # 112 * 112 * 32 -> 112 * 112 * 64
            conv_dw(64, 128, 2),   # 112 * 112 * 64 -> 56 * 56 * 128
            conv_dw(128, 128, 1),  # 56 * 56 * 128 -> 56 * 56 * 128
            conv_dw(128, 256, 2),  # 56 * 56 * 128 -> 28 * 28 * 256
            conv_dw(256, 256, 1),  # 28 * 28 * 256 -> 28 * 28 * 256
            conv_dw(256, 512, 2),  # 28 * 28 * 256 -> 14 * 14 * 512
            conv_dw(512, 512, 1),  # 14 * 14 * 512 -> 14 * 14 * 512
            conv_dw(512, 512, 1),  # 14 * 14 * 512 -> 14 * 14 * 512
            conv_dw(512, 512, 1),  # 14 * 14 * 512 -> 14 * 14 * 512
            conv_dw(512, 512, 1),  # 14 * 14 * 512 -> 14 * 14 * 512
            conv_dw(512, 512, 1),  # 14 * 14 * 512 -> 14 * 14 * 512
            conv_dw(512, 1024, 2), # 14 * 14 * 512 -> 7 * 7 * 1024
            conv_dw(1024, 1024, 1), # 7 * 7 * 1024 -> 7 * 7 * 1024
            nn.AdaptiveAvgPool2d(1) # 7 * 7 * 1024 -> 1 * 1 * 1024
        )
        self.fc = nn.Linear(1024, n_classes)


    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



if __name__=='__main__':
    # model check
    model = MobileNetV1(ch_in=3, n_classes=1000)
    summary(model, input_size=(3, 224, 224), device='cpu')
    from thop import profile
    input_size = (1, 3, 224, 224)
    flops, params = profile(model=model, input_size=input_size)
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))