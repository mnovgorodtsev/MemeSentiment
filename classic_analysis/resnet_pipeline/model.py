import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from classic_analysis.base import MultiTaskModel


class ResNetLinear(MultiTaskModel):
    def __init__(self, base_model=None, tasks=None, dropout=0.1):
        super().__init__(tasks)
        self.base = base_model or resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(in_features, 2) for task in self.tasks}
        )

    def forward(self, images):
        features = self.base(images)
        features = self.dropout(features)
        return {task: self.heads[task](features) for task in self.tasks}


class ResNetAttention(MultiTaskModel):
    def __init__(self, base_model=None, tasks=None, dropout=0.2):
        super().__init__(tasks)
        self.base = base_model or resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid(),
        )

        self.heads = nn.ModuleDict(
            {task: nn.Linear(in_features, 2) for task in self.tasks}
        )

    def forward(self, images):
        features = self.base(images)
        features = self.dropout(features)

        attention = self.channel_attention(features)
        features_attended = features * attention

        return {task: self.heads[task](features_attended) for task in self.tasks}


class ResNetAdaptivePooling(MultiTaskModel):
    def __init__(self, base_model=None, tasks=None, dropout=0.2, hidden_dim=256):
        super().__init__(tasks)
        self.base = base_model or resnet18(weights=ResNet18_Weights.DEFAULT)

        self.base.fc = nn.Identity()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        spatial_features = 512 * 4 * 4

        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict()
        for task in self.tasks:
            self.heads[task] = nn.Sequential(
                nn.Linear(spatial_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2),
            )

    def forward(self, images):
        x = images
        for name, module in self.base.named_children():
            if name == "fc":
                break
            x = module(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return {task: self.heads[task](x) for task in self.tasks}
