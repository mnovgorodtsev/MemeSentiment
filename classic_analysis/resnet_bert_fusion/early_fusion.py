import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from classic_analysis.datasets_preparation import _load_and_split_data, FusionDataset
from classic_analysis.base import MultiTaskModel
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class EarlyFusionModel(MultiTaskModel):

    def __init__(self, dropout=0.2):
        super().__init__()

        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        img_features = base.fc.in_features
        base.fc = nn.Identity()
        self.image_encoder = base
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        txt_features = self.text_encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        fused_dim = img_features + txt_features

        self.heads = nn.ModuleDict({
            task: nn.Linear(fused_dim, 2)
            for task in self.tasks
        })

    def forward(self, images, input_ids, attention_mask):

        img_features = self.image_encoder(images)

        txt_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        txt_features = txt_out.last_hidden_state[:, 0]

        img_features = self.dropout(img_features)
        txt_features = self.dropout(txt_features)

        # Early fusion
        fused = torch.cat([img_features, txt_features], dim=1)

        return {
            task: self.heads[task](fused)
            for task in self.tasks
        }


class EarlyFusionTrainer:

    def __init__(
        self,
        csv_path,
        images_dir,
        save_path="./models/early_fusion_model/model_weights.pt",
        batch_size=16,
        epochs=10,
        lr=2e-5,
        dropout=0.2,
        val_split=0.1
    ):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"


        self.tasks = ["humour", "sarcasm", "offensive", "motivational"]
        self.epochs = epochs
        self.save_path = save_path

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        train_df, test_df, _ = _load_and_split_data(csv_path)
        full_train = FusionDataset(train_df, images_dir, self.tokenizer, self.transform)
        val_size = int(len(full_train) * val_split)
        train_size = len(full_train) - val_size
        train_dataset, val_dataset = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )


        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            FusionDataset(test_df, images_dir, self.tokenizer, self.transform),
            batch_size=batch_size,
            shuffle=False
        )

        self.model = EarlyFusionModel(dropout=dropout).to(self.device)

        self.criterions = {}
        for task in self.tasks:
            weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df[task])
            w = torch.tensor(weights, dtype=torch.float).to(self.device)
            self.criterions[task] = nn.CrossEntropyLoss(weight=w)

        self.optimizer = AdamW([
            {"params": self.model.text_encoder.parameters(), "lr": 2e-5},
            {"params": self.model.image_encoder.parameters(), "lr": 1e-4},
            {"params": self.model.heads.parameters(), "lr": 1e-4},
        ], weight_decay=1e-2)

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )

    def _train_epoch(self):

        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            self.optimizer.zero_grad()

            logits = self.model(images, input_ids, attention_mask)

            loss = sum(
                self.criterions[task](logits[task], labels[task].to(self.device))
                for task in self.tasks
            )

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _evaluate(self, loader):

        self.model.eval()

        y_true = {task: [] for task in self.tasks}
        y_pred = {task: [] for task in self.tasks}
        total_loss = 0.0

        with torch.no_grad():

            for batch in loader:

                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                logits = self.model(images, input_ids, attention_mask)

                loss = sum(
                    self.criterions[task](logits[task], labels[task].to(self.device))
                    for task in self.tasks
                )
                total_loss += loss.item()

                for task in self.tasks:
                    preds = torch.argmax(logits[task], dim=1)
                    y_pred[task].extend(preds.cpu().numpy())
                    y_true[task].extend(labels[task].cpu().numpy())

        avg_loss = total_loss / len(loader)
        avg_acc = sum(
            accuracy_score(y_true[t], y_pred[t]) for t in self.tasks
        ) / len(self.tasks)

        return avg_loss, avg_acc, y_true, y_pred

    def train(self):

        best_val_acc = 0.0

        for epoch in range(1, self.epochs + 1):

            train_loss = self._train_epoch()
            val_loss, val_acc, _, _ = self._evaluate(self.val_loader)
            self.scheduler.step()

            print(
                f"Epoch {epoch:02d}/{self.epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  → Saved model (val_acc: {val_acc:.4f})")

        print(f"\nBest val_acc: {best_val_acc:.4f}")

    def evaluate_test(self):

        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device)
        )

        _, _, y_true, y_pred = self._evaluate(self.test_loader)

        for task in self.tasks:

            print(f"\n{task.upper()}")
            print("Accuracy:", accuracy_score(y_true[task], y_pred[task]))
            print(
                classification_report(
                    y_true[task],
                    y_pred[task],
                    digits=4,
                    zero_division=0
                )
            )
            print("Confusion matrix:")
            print(confusion_matrix(y_true[task], y_pred[task]))


class EarlyFusionEvaluator:
    def __init__(
        self,
        csv_path,
        images_dir,
        model_path,
        batch_size=16
    ):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.tasks = ["humour", "sarcasm", "offensive", "motivational"]

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # _load_and_split_data zwraca (train_df, test_df, encoders)
        _, test_df, _ = _load_and_split_data(csv_path)

        self.loader = DataLoader(
            FusionDataset(test_df, images_dir, self.tokenizer, self.transform),
            batch_size=batch_size,
            shuffle=False
        )

        self.model = EarlyFusionModel().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

    def evaluate(self):

        y_true = {task: [] for task in self.tasks}
        y_pred = {task: [] for task in self.tasks}

        with torch.no_grad():

            for batch in self.loader:

                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                logits = self.model(images, input_ids, attention_mask)

                for task in self.tasks:
                    preds = torch.argmax(logits[task], dim=1)
                    y_pred[task].extend(preds.cpu().numpy())
                    y_true[task].extend(labels[task].cpu().numpy())

        for task in self.tasks:

            print(f"\n{task.upper()}")
            print("Accuracy:", accuracy_score(y_true[task], y_pred[task]))
            print(
                classification_report(
                    y_true[task],
                    y_pred[task],
                    digits=4,
                    zero_division=0
                )
            )
            print("Confusion matrix:")
            print(confusion_matrix(y_true[task], y_pred[task]))


if __name__ == "__main__":
    trainer = EarlyFusionTrainer(
        csv_path="data/memotion_dataset_7k/labels.csv",
        images_dir="data/memotion_dataset_7k/images",
        save_path="./models/early_fusion_model/model_weights.pt",
        batch_size=16,
        epochs=10,
        lr=2e-5,
        dropout=0.3
    )
    trainer.train()
    trainer.evaluate_test()

    # evaluator = EarlyFusionEvaluator(
    #     csv_path="data/memotion_dataset_7k/labels.csv",
    #     images_dir="data/memotion_dataset_7k/images",
    #     model_path="./models/early_fusion_model/model_weights.pt",
    #     batch_size=16
    # )
    # evaluator.evaluate()