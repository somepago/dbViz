'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import timm

class ViT_pt_interpolate(nn.Module):
    def __init__(self, pretrained=True):
        super(ViT_pt_interpolate, self).__init__()
        self.net = timm.create_model("vit_small_patch16_224", pretrained=pretrained)
        self.net.head = nn.Linear(self.net.head.in_features, 10)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224))
        out = self.net(x)
        return out
