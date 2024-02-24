"""
Count the MACs/FLOPs/parameters of PyTorch models.
Installation: pip install flopco-pytorch
"""
from flopco import FlopCo
from torchvision.models import resnet50
from networks.UNet import U_Net
from networks.RAUNet import RAUNet
from networks.NestedUNet import NestedUNet
from networks.deeplabv3_plus import DeepLabV3Plus
from networks.PspNet.pspnet import PSPNet

channels = 2
device = 'cuda'
model = PSPNet(num_classes=2).to(device)

# Estimate model statistics by making one forward pass througth the model,
# for the input image of size 3 x 224 x 224

stats = FlopCo(model, img_size=(2, 1, 384, 384), device=device)

print(stats.total_macs, stats.total_flops)
