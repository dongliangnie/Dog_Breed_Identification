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
    
class EnsembleSoftVoting(torch.nn.Module):
    def __init__(self, model1, model2, weight1=0.5, weight2=0.5):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.w1 = weight1
        self.w2 = weight2

    def forward(self, x):
        # 输出 logits
        logits1 = self.model1(x)
        logits2 = self.model2(x)

        # softmax 变成概率再加权
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)

        # 加权融合
        final_probs = self.w1 * probs1 + self.w2 * probs2

        return final_probs

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


