import torch
from torch import nn
import warnings
import torchvision.models as models

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

class SwAVFeatures(nn.Module):
    def __init__(self):
        super(SwAVFeatures, self).__init__()
        model = torch.hub.load('facebookresearch/swav:main', 'resnet50')

        self.features = torch.nn.Sequential(*list(model.children())[:-1])
    
        self.features.eval()

        for param in self.features.parameters():
            param.requires_grad = False
    
    # 2048 dimensional features
    def forward(self, x):
        return self.features(x)


class DINO_ResNet50_Features(nn.Module):
    def __init__(self):
        super(DINO_ResNet50_Features, self).__init__()
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

        self.features = torch.nn.Sequential(*list(model.children())[:-1])
    
        self.features.eval()

        for param in self.features.parameters():
            param.requires_grad = False
    
    # 2048 dimensional features
    def forward(self, x):
        return self.features(x)
    

class DINO_ViTs16_Features(nn.Module):
    def __init__(self):
        super(DINO_ViTs16_Features, self).__init__()
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

        self.features = model
    
        self.features.eval()

        for param in self.features.parameters():
            param.requires_grad = False

    # 384 dimensional features
    def forward(self, x):
        return self.features(x)


class DINO_ViTs8_Features(nn.Module):
    def __init__(self):
        super(DINO_ViTs8_Features, self).__init__()
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

        self.features = model
    
        self.features.eval()

        for param in self.features.parameters():
            param.requires_grad = False

    # 384 dimensional features
    def forward(self, x):
        return self.features(x)


class ResNet18Features(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Features, self).__init__()
    
        model = models.resnet18(pretrained=pretrained)
        
        self.features = nn.Sequential(*list(model.children())[:-1])
       
        self.features.eval()

        for param in self.features.parameters():
            param.requires_grad = False
    
    # 512 dimensional features
    def forward(self, x):
        return self.features(x)
