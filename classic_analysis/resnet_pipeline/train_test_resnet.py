import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from classic_analysis.base import MultiTaskModel, MultiTaskTrainer
from torchvision.models import resnet18, ResNet18_Weights


class ResNetMultiTaskTrainer(MultiTaskModel):
    def __init__(self, base_model=None, tasks=None):
        super().__init__(tasks)
        self.base = base_model or resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.heads = nn.ModuleDict({task: nn.Linear(in_features, 2) for task in self.tasks})

    def forward(self, x):
        features = self.base(x)
        return {task: self.heads[task](features) for task in self.tasks}


if __name__ == "__main__":
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = ResNetMultiTaskTrainer(base_model=base_model)
    trainer = MultiTaskTrainer(
        model=model,
        csv_path="data/memotion_dataset_7k/labels.csv",
        data_type="image",
        images_dir="data/memotion_dataset_7k/images",
        save_path="./models/resnet_multitask_model",
        batch_size=32,
        epochs=10,
        lr_finetune=1e-4,
        freeze_epochs=3,
        test=True
    )

    # trainer.train()
    trainer.test()
