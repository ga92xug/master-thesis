import torch
from typing import Any, Optional
import warnings
from torchvision import models
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
warnings.filterwarnings("ignore", message="pytorch_quantization module")

import os
import sys
sys.path.append(f"{os.getcwd()}")

from src.networks.compare_models.vision_transformer import vit_b_16

def pretrained_weights(pre_trained: bool):    
    if pre_trained:
        weights='IMAGENET1K_V1'
    else:
        weights=None
    return weights


class EfficientNet(torch.nn.Module):
    """
    We could have used all models from torchvision.models. The b0 model is from NVIDIA's DeepLearningExamples
    """

    def __init__(self, num_classes, num_channels: int = 3, pre_trained=True, size="b0", **kwargs):
        super().__init__()
        if size == "b0" and False:
            model_name='nvidia_efficientnet_' + size
            self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', model_name, pretrained=pre_trained)
            # Modify the last fully connected layer to have num_classes outputs
            in_features = self.model.classifier[3].in_features  # The 'fc' layer is the 4th layer in 'classifier', so its index is 3
            self.model.classifier[3] = torch.nn.Linear(in_features, num_classes)
        else:
            # issue with hash of pretrained weights
            # https://github.com/mrdbourke/pytorch-deep-learning/issues/696#issuecomment-1776124098
            def get_state_dict(self, *args, **kwargs):
                kwargs.pop("check_hash")
                return load_state_dict_from_url(self.url, *args, **kwargs)
            WeightsEnum.get_state_dict = get_state_dict

            self.model = getattr(models, f"efficientnet_{size}")(weights=pretrained_weights(pre_trained))
            # Modify the last fully connected layer to have num_classes outputs
            self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, num_classes, bias=True)

            if num_channels != 3:
                from torchvision.ops.misc import Conv2dNormActivation
                from torchvision.models.efficientnet import _efficientnet_conf
                # Modify the first layer to have num_channels inputs
                firstconv_output_channels = _efficientnet_conf(f"efficientnet_b{size}", width_mult=1.0, depth_mult=1.0)[0][0].input_channels
                self.model.features[0] = Conv2dNormActivation(
                    num_channels, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=torch.nn.BatchNorm2d, activation_layer=torch.nn.SiLU
                )

        self.name = "EfficientNet_pre_imagenet" if pre_trained else "EfficientNet"


    def forward(self, x):
        return self.model(x)
    
class ViT(torch.nn.Module):
    def __init__(self, num_classes, num_channels: int = 3, pre_trained=True, image_size=224, **kwargs):
        super().__init__()
        self.model = vit_b_16(weights=pretrained_weights(pre_trained), image_size=image_size)

        if num_channels != 3:
            self.model.conv_proj = torch.nn.Conv2d(
                in_channels=num_channels, out_channels=768, kernel_size=16, stride=16
            )


        # Modify the last fully connected layer to have num_classes
        in_features = self.model.heads[0].in_features
        self.model.heads[0] = torch.nn.Linear(in_features, num_classes)

        self.name = "ViT_pre_imagenet" if pre_trained else "ViT"

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    image_size = 128
    num_channels = 1
    #model = EfficientNet(size="b0", num_classes=8, pretrained=True)
    model = ViT(num_classes=10, num_channels=num_channels, pre_trained=True, image_size=image_size)
    print(model)
    x = torch.rand(1, num_channels, image_size, image_size)
    out = model(x)
    print(out.shape)
    print(out)