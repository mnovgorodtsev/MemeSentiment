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

from classic_analysis.datasets_preparation import _load_and_split_data, ImageHumourDataset


class ResnetHumourTrainer:
    def __init__(
        self,
        csv_path,
        images_dir,
        save_path="./resnet_model/resnet_model.pth",
        batch_size=32,
        epochs=10,
        lr_head=1e-3,
        lr_finetune=1e-4,
        freeze_epochs=3,
        num_workers=2,
        test=False,
        test_size=0.2,
        random_state=42,
    ):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.save_path = save_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_head = lr_head
        self.lr_finetune = lr_finetune
        self.freeze_epochs = freeze_epochs
        self.num_workers = num_workers
        self.test_size = test_size
        self.random_state = random_state
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        self.train_df, self.test_df, _ = _load_and_split_data(
            csv_path, test_size=self.test_size, random_state=self.random_state
        )

        if test:
            self.load_model()

    def build_model(self):
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)

        # freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # train only head
        for param in model.fc.parameters():
            param.requires_grad = True

        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.fc.parameters(),
            lr=self.lr_head
        )

    def unfreeze_last_block(self):
        print(">>> Unfreezing layer4 for fine-tuning")
        for name, param in self.model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr_finetune
        )

    def build_loaders(self):
        train_dataset = ImageHumourDataset(self.train_df, self.images_dir, self.transform)
        test_dataset = ImageHumourDataset(self.test_df, self.images_dir, self.transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return train_loader, test_loader

    def train(self):
        self.build_model()
        train_loader, _ = self.build_loaders()

        for epoch in range(self.epochs):
            if epoch == self.freeze_epochs:
                self.unfreeze_last_block()

            self.model.train()
            total_loss = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")

        self.save_model()

    def save_model(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_model(self):
        self.build_model()
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from {self.save_path}")

    def test(self):
        _, test_loader = self.build_loaders()

        y_true = []
        y_pred = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        self.metrics(y_true, y_pred)

    @staticmethod
    def metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}\n")

        print(classification_report(
            y_true, y_pred,
            target_names=["not_funny", "funny"],
            digits=4,
            zero_division=0
        ))

        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    trainer = ResnetHumourTrainer(
        csv_path="../../data/memotion_dataset_7k/new_labels.csv",
        images_dir="../../data/memotion_dataset_7k/images",
        epochs=10,
        batch_size=32
    )

    trainer.train()
