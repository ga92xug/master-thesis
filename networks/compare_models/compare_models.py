import torch
from torchvision.models import vit_l_16

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
    def __init__(self, num_classes, pretrained=True, **kwargs):
        super().__init__()
        if pretrained:
            weights='IMAGENET1K_V1'
        else:
            weights=None
        self.model = vit_l_16(weights=weights)
        # Modify the last fully connected layer to have num_classes
        in_features = self.model.heads[0].in_features
        self.model.heads[0] = torch.nn.Linear(in_features, num_classes)

        self.name = "ViT_pre_imagenet" if pretrained else "ViT"

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    #model = EfficientNet()
    model = ViT(num_classes=10, pretrained=True)
    print(model)
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    print(out.shape)
    print(out)