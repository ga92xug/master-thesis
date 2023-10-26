import torch
from typing import Any, Optional
import warnings
warnings.filterwarnings("ignore", message="pytorch_quantization module")

import os
import sys
sys.path.append(f"{os.getcwd()}")

from networks.compare_models.vision_transformer import vit_b_16

class EfficientNet(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True, **kwargs):
        super().__init__()
        model_name='nvidia_efficientnet_b0'
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', model_name, pretrained=pretrained)
        # Modify the last fully connected layer to have num_classes outputs
        in_features = self.model.classifier[3].in_features  # The 'fc' layer is the 4th layer in 'classifier', so its index is 3
        self.model.classifier[3] = torch.nn.Linear(in_features, num_classes)

        self.name = "EfficientNet_pre_imagenet" if pretrained else "EfficientNet"

    def forward(self, x):
        return self.model(x)
    
class ViT(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True, image_size=224, **kwargs):
        super().__init__()
        if pretrained:
            weights='IMAGENET1K_V1'

        else:
            weights=None

        self.model = vit_b_16(weights=weights, image_size=image_size)

        # Modify the last fully connected layer to have num_classes
        in_features = self.model.heads[0].in_features
        self.model.heads[0] = torch.nn.Linear(in_features, num_classes)

        self.name = "ViT_pre_imagenet" if pretrained else "ViT"

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    image_size = 128
    #model = EfficientNet()
    #_vision_transformer()
    model = ViT(num_classes=10, pretrained=False, image_size=image_size)
    print(model)
    x = torch.rand(1, 3, image_size, image_size)
    out = model(x)
    print(out.shape)
    print(out)