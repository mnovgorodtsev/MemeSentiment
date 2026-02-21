import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from classic_analysis.datasets_preparation import _load_and_split_data, ImageMultiTaskDataset


class ResNetMultiTask(nn.Module):
    def __init__(self, base_model=None, tasks=None):
        super().__init__()
        self.tasks = tasks or ["humour", "sarcasm", "offensive", "motivational"]

        self.base = base_model or resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity()

        self.heads = nn.ModuleDict({
            task: nn.Linear(in_features, 2) for task in self.tasks
        })

    def forward(self, x):
        features = self.base(x)
        outputs = {task: self.heads[task](features) for task in self.tasks}
        return outputs


class ResNetMultiTaskTrainer:
    def __init__(self, 
                 csv_path, 
                 images_dir, 
                 save_path="./resnet_multitask_model.pth", 
                 batch_size=32, 
                 epochs=10, 
                 lr_head=1e-3, 
                 lr_finetune=1e-4, 
                 freeze_epochs=3, 
                 num_workers=2,
                 test=False):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_path = save_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_head = lr_head
        self.lr_finetune = lr_finetune
        self.freeze_epochs = freeze_epochs
        self.num_workers = num_workers
        self.images_dir = images_dir
        self.test_size = 0.2
        self.random_state = 42

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        self.tasks = ["humour", "sarcasm", "offensive", "motivational"]
        self.model = ResNetMultiTask(resnet18(weights=ResNet18_Weights.DEFAULT), tasks=self.tasks).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.heads.parameters(),
            lr=self.lr_head
        )

        self.train_df, self.test_df, _ = _load_and_split_data(csv_path)

        if test:
            self.load_model()

    def build_loaders(self):
        train_dataset = ImageMultiTaskDataset(self.train_df, self.images_dir, self.transform)
        test_dataset = ImageMultiTaskDataset(self.test_df, self.images_dir, self.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, test_loader

    def train(self):
        train_loader, _ = self.build_loaders()

        for epoch in range(self.epochs):
            if epoch == self.freeze_epochs:
                print(">>> Unfreezing layer4")
                for name, param in self.model.base.named_parameters():
                    if "layer4" in name:
                        param.requires_grad = True
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.lr_finetune
                )

            self.model.train()
            total_loss = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                for task in self.tasks:
                    labels[task] = labels[task].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = sum(self.criterion(outputs[task], labels[task]) for task in self.tasks)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        self.save_model()

    def save_model(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_model(self):
        state_dict = torch.load(self.save_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Loaded model from {self.save_path}")

    def test(self):
        _, test_loader = self.build_loaders()
        self.model.eval()

        y_true = {task: [] for task in self.tasks}
        y_pred = {task: [] for task in self.tasks}

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                for task in self.tasks:
                    labels[task] = labels[task].to(self.device)

                outputs = self.model(images)
                for task in self.tasks:
                    preds = torch.argmax(outputs[task], dim=1)
                    y_true[task].extend(labels[task].cpu().numpy())
                    y_pred[task].extend(preds.cpu().numpy())

        for task in self.tasks:
            self.metrics(y_true, y_pred, task)

    @staticmethod
    def metrics(y_true, y_pred, task):
        print(f"\n===== {task.upper()} =====")
        acc = accuracy_score(y_true[task], y_pred[task])
        precision, recall, f1, _ = precision_recall_fscore_support(y_true[task], 
                                                                   y_pred[task], 
                                                                   average="weighted")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}\n")

        print(classification_report(y_true[task], 
                                    y_pred[task], 
                                    digits=4, 
                                    zero_division=0))
        
        print(confusion_matrix(y_true[task], y_pred[task]))


if __name__ == "__main__":
    trainer = ResNetMultiTaskTrainer(
        csv_path="data/memotion_dataset_7k/labels.csv",
        images_dir="data/memotion_dataset_7k/images",
        epochs=10,
        batch_size=32,
        test=True
    )

    # trainer.train()
    trainer.test()
