import torch
import torch.nn as nn
import torch.nn.functional as F
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # 去掉 fc 层，只保留 conv1 → layer4
        self.features = nn.Sequential(*list(backbone.children())[:-1])  
        self.out_dim = backbone.fc.in_features

    def forward(self, x):
        x = self.features(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)
        return x
class SENetFeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # 去掉最后的 fc
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = backbone.last_linear.in_features

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

class FeatureFusionEnsemble(nn.Module):
    """
    两个模型输出 feature，进行 concat 后接新的分类头
    """
    def __init__(self, feat_extractor1, feat_extractor2, num_classes):
        super().__init__()
        self.m1 = feat_extractor1
        self.m2 = feat_extractor2

        fusion_dim = feat_extractor1.out_dim + feat_extractor2.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f1 = self.m1(x)
        f2 = self.m2(x)

        fusion = torch.cat([f1, f2], dim=1)

        logits = self.classifier(fusion)
        return logits, fusion


