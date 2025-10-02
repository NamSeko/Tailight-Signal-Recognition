import torch.nn as nn
from torchvision import models

class TaillightClassification(nn.Module):
    def __init__(self, num_classes=3):
        super(TaillightClassification, self).__init__()
        self.num_classes = num_classes
        # ResNet18
        # self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # self.feature_dim = self.backbone.fc.in_features
        # self.backbone.fc = nn.Identity()
     
        # RegNet_Y_1_6GF
        # self.backbone = models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2)
        # self.feature_dim = self.backbone.fc.in_features
        # self.backbone.fc = nn.Identity()
        
        # RegNet_Y_8GF
        self.backbone = models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V2)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.feature_dim, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x