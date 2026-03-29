import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from classic_analysis.base import MultiTaskModel
from torchvision.models import resnet18, ResNet18_Weights


class ResNetMultiTaskModel(MultiTaskModel):
    def __init__(self, base_model=None, tasks=None):
        super().__init__(tasks)
        self.base = base_model or resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.heads = nn.ModuleDict({task: nn.Linear(in_features, 2) for task in self.tasks})

    def forward(self, x):
        features = self.base(x)
        return {task: self.heads[task](features) for task in self.tasks}
