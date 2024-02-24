import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from networks.mynet2.FSM import FSM
from networks.mynet2.branch_SwinB import SwinT


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class Net2(nn.Module):
    def __init__(
            self,
            class_num=2,
            encoder_channels=None,
            pyramid_channels=256,
            segmentation_channels=64,
            dropout=0.2,
    ):
        super().__init__()
        # ==> transformer encoder layers
        if encoder_channels is None:
            encoder_channels = [512, 256, 128, 64]

        self.swint = SwinT()
        # ==> CNN encoder layers
        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.layer_down0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer_down1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer_down2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer_down3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer_down4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])  # 256-256
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])  # 256-128
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])  # 256-64

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.final_conv = nn.Conv2d(segmentation_channels, class_num, kernel_size=1, padding=0)

        self.mid5_conv = nn.Conv2d(encoder_channels[0], 2, kernel_size=1, padding=0)
        self.mid4_conv = nn.Conv2d(encoder_channels[1], 2, kernel_size=1, padding=0)
        self.mid3_conv = nn.Conv2d(encoder_channels[2], 2, kernel_size=1, padding=0)
        self.FSM3 = FSM(inplanes=128)
        self.FSM4 = FSM(inplanes=256)
        self.FSM5 = FSM(inplanes=512)

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.cat([x, x, x], 1)
        x0 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        x2_2, x3_2, x4_2, x4, x3, x2 = self.swint(x0)
        # print("pred", x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        # ==> get encoder features
        c1 = self.layer_down0(x)
        c2 = self.layer_down1(c1)
        c3 = self.layer_down2(c2)  # 128
        # print(c3.shape)
        # 特征选择模块
        c3 = self.FSM3(c3, x2_2)
        c4 = self.layer_down3(c3)  # 256
        c4 = self.FSM4(c4, x3_2)
        c5 = self.layer_down4(c4)  # 512
        c5 = self.FSM5(c5, x4_2)

        # 语义蒸馏损失
        a5 = self.mid5_conv(c5)
        a5 = F.interpolate(a5, size=(384, 384), mode='bilinear', align_corners=True)
        a4 = self.mid4_conv(c4)
        a4 = F.interpolate(a4, size=(384, 384), mode='bilinear', align_corners=True)
        a3 = self.mid3_conv(c3)
        a3 = F.interpolate(a3, size=(384, 384), mode='bilinear', align_corners=True)

        # print("c5", c5.shape)
        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])
        # print("p", p5.shape, p4.shape, p3.shape, p2.shape)

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)
        # print("s", s5.shape, s4.shape, s3.shape, s2.shape)
        x = s5 + s4 + s3 + s2
        x = self.dropout(x)

        x = self.final_conv(x)
        # print(x.shape)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x, a5, a4, a3,  x4, x3, x2


if __name__ == "__main__":
    model = Net2()
    x = torch.randn(1, 1, 384, 384)
    out = model(x)
    print(out[1].shape)
    from torchinfo import summary

    summary(model, x.shape)
