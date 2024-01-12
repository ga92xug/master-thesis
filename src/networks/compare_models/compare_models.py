import torch
from typing import Any, Optional
import warnings
from torchvision import models
warnings.filterwarnings("ignore", message="pytorch_quantization module")

import os
import sys
sys.path.append(f"{os.getcwd()}")

from src.networks.compare_models.vision_transformer import vit_b_16

def pretrained_weights(pretrained: bool):    
    if pretrained:
        weights='IMAGENET1K_V1'
    else:
        weights=None
    return weights


class EfficientNet(torch.nn.Module):
    """
    We could have used all models from torchvision.models. The b0 model is from NVIDIA's DeepLearningExamples
    """

    def __init__(self, num_classes, pretrained=True, size="b0", **kwargs):
        super().__init__()
        if size == "b0":
            model_name='nvidia_efficientnet_' + size
            self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', model_name, pretrained=pretrained)
            # Modify the last fully connected layer to have num_classes outputs
            in_features = self.model.classifier[3].in_features  # The 'fc' layer is the 4th layer in 'classifier', so its index is 3
            self.model.classifier[3] = torch.nn.Linear(in_features, num_classes)
        else:
            if pretrained:

                raise NotImplementedError("Pretrained EfficientNet models are not available in torchvision, at the moment get invalid hash value")

            self.model = getattr(models, f"efficientnet_{size}")(weights=pretrained_weights(pretrained))
            # Modify the last fully connected layer to have num_classes outputs
            self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, num_classes, bias=True)

        self.name = "EfficientNet_pre_imagenet" if pretrained else "EfficientNet"


    def forward(self, x):
        return self.model(x)
    
class ViT(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True, image_size=224, **kwargs):
        super().__init__()
        self.model = vit_b_16(weights=pretrained_weights(pretrained), image_size=image_size)

        # Modify the last fully connected layer to have num_classes
        in_features = self.model.heads[0].in_features
        self.model.heads[0] = torch.nn.Linear(in_features, num_classes)

        self.name = "ViT_pre_imagenet" if pretrained else "ViT"

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    image_size = 528
    model = EfficientNet(size="b6", num_classes=8, pretrained=False)
    #model = ViT(num_classes=10, pretrained=False, image_size=image_size)
    print(model)
    x = torch.rand(1, 3, image_size, image_size)
    out = model(x)
    print(out.shape)
    print(out)