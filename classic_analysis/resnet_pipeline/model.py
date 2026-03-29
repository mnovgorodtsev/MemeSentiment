import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from classic_analysis.base import MultiTaskModel


class ResNetMultiTaskModel(MultiTaskModel):
    def __init__(self, base_model=None, tasks=None):
        super().__init__(tasks)
        self.base = base_model or resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.heads = nn.ModuleDict(
            {task: nn.Linear(in_features, 2) for task in self.tasks}
        )

    def forward(self, x):
        features = self.base(x)
        return {task: self.heads[task](features) for task in self.tasks}
