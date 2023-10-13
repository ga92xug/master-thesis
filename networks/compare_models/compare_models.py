import torch
from torchvision.models import vit_l_16

class EfficientNet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model_name='nvidia_efficientnet_b0'
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', model_name, pretrained=pretrained)

    def forward(self, x):
        return self.model(x)
    
class ViT(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights='IMAGENET1K_V1'
        else:
            weights=None
        self.model = vit_l_16(weights=weights)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    #model = EfficientNet()
    model = ViT()
    print(model)
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    print(out.shape)
    print(out)