import numpy as np

from .tranception.tansception import Transception
from .Transunet.vit_seg_modeling import VisionTransformer, CONFIGS
from .swinunet.vision_transformer import SwinUnet

vit_name = "R50-ViT-B_16"
config_vit = CONFIGS['R50-ViT-B_16']
config_vit.n_classes = 2
config_vit.n_skip = 3
config_vit.patches.grid = (int(224 / 16), int(224 / 16))
from .DAENet.DAENet import DAEFormer
from .mynet2.Net2 import Net2

def get_model(model_name: str, channels: int):
    assert model_name.lower() in ["transunet", "swinunet", "transception", "dae",
                                  "mynet2", "a2fpn"
                                  ]
    if model_name.lower() == 'dae':
        model = DAEFormer()
    elif model_name.lower() == 'transception':
        model = Transception(num_classes=2, head_count=8, dil_conv=1, token_mlp_mode="mix_skip", concat='sk')
    elif model_name.lower() == 'transunet':
        model = VisionTransformer(config_vit, img_size=224, num_classes=2)
        model.load_from(weights=np.load(config_vit.pretrained_path))
    elif model_name.lower() == 'swinunet':
        model = SwinUnet(img_size=224, num_classes=2, zero_head=False, vis=False)
        model.load_from()
    elif model_name.lower() == 'mynet2':
        model = Net2()
    return model
