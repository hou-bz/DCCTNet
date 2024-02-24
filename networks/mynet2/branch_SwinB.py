import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from networks.mynet2.CFM import BasicConv2d
from networks.mynet2.SwinTransformer import SwinB_224


class SwinT(nn.Module):
    def __init__(self, out_ch=2, segmentation_channels=64, pretrained=True):
        super(SwinT, self).__init__()
        # self.backbone = self.backbone = SwinB_384(pretrained=pretrained)
        self.backbone = self.backbone = SwinB_224(pretrained=pretrained)
        filter = [96, 192, 384, 768]

        self.conv_128 = BasicConv2d(filter[1], 128, 1)
        self.conv_256 = BasicConv2d(filter[2], 256, 1)
        self.conv_512 = BasicConv2d(filter[3], 512, 1)
        self.up1 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=out_ch, kernel_size=1, stride=1),
                                 nn.Upsample(size=(384, 384)))
        self.up2 = nn.Sequential(nn.Conv2d(in_channels=filter[1], out_channels=out_ch, kernel_size=1, stride=1),
                                 nn.Upsample(size=(384, 384)))
        self.up3 = nn.Sequential(nn.Conv2d(in_channels=filter[2], out_channels=out_ch, kernel_size=1, stride=1),
                                 nn.Upsample(size=(384, 384)))
        self.up4 = nn.Sequential(nn.Conv2d(in_channels=filter[3], out_channels=out_ch, kernel_size=1, stride=1),
                                 nn.Upsample(size=(384, 384)))
        self.up5 = nn.Sequential(nn.Conv2d(in_channels=filter[3], out_channels=out_ch, kernel_size=1, stride=1),
                                 nn.Upsample(size=(384, 384)))

    def forward(self, x):
        B, _, H, W = x.shape  # (b, 32H, 32W, 3)

        x1 = self.backbone.stem(x)  # 8h 8w  位置信息
        x2 = self.backbone.layers[0](x1)  # 4h 4w
        x3 = self.backbone.layers[1](x2)  # 2h 2w
        x4 = self.backbone.layers[2](x3)  # h w
        x5 = self.backbone.layers[3](x4)  # hw

        x1 = x1.view(B, H // 4, W // 4, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.view(B, H // 8, W // 8, -1).permute(0, 3, 1, 2).contiguous()
        x3 = x3.view(B, H // 16, W // 16, -1).permute(0, 3, 1, 2).contiguous()
        x4 = x4.view(B, H // 32, W // 32, -1).permute(0, 3, 1, 2).contiguous()
        x5 = x5.view(B, H // 32, W // 32, -1).permute(0, 3, 1, 2).contiguous()
        # print("feature", x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        # feature torch.Size([1, 96, 56, 56]) torch.Size([1, 192, 28, 28])
        # torch.Size([1, 384, 14, 14]) torch.Size([1, 768, 7, 7]) torch.Size([1, 768, 7, 7])

        # print("feature", x2.shape, x3.shape, x4.shape)
        # feature torch.Size([1, 64, 28, 28]) torch.Size([1, 64, 14, 14]) torch.Size([1, 64, 7, 7])
        x2_2 = self.conv_128(x2)
        x2_2 = F.interpolate(x2_2, size=(48, 48), mode='bilinear', align_corners=True)
        x3_2 = self.conv_256(x3)
        x3_2 = F.interpolate(x3_2, size=(24, 24), mode='bilinear', align_corners=True)
        x4_2 = self.conv_512(x4)
        x4_2 = F.interpolate(x4_2, size=(12, 12), mode='bilinear', align_corners=True)

        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = self.up4(x4)

        # print("pred",x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        return x2_2, x3_2, x4_2, x4, x3, x2


if __name__ == "__main__":
    model = SwinT(out_ch=2)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out[0].shape)
    from torchinfo import summary

    summary(model, x.shape)
