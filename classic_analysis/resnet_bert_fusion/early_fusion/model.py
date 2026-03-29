import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel, AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight

from classic_analysis.base import MultiTaskModel, print_task_metrics
from classic_analysis.base.helpers import compute_mean_std, save_results_csv
from classic_analysis.datasets_preparation import FusionDataset, _load_and_split_data


def early_fusion_unpack(batch: dict, device: str) -> tuple[dict, dict]:
    inputs = {
        "images":         batch["image"].to(device),
        "input_ids":      batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    labels = {task: batch["labels"][task] for task in batch["labels"]}
    return inputs, labels


class EarlyFusionModel(MultiTaskModel):
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()

        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        img_features = base.fc.in_features
        base.fc = nn.Identity()
        self.image_encoder = base

        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        txt_features = self.text_encoder.config.hidden_size

        self.dropout  = nn.Dropout(dropout)
        fused_dim = img_features + txt_features

        self.heads = nn.ModuleDict({
            task: nn.Linear(fused_dim, 2) for task in self.tasks
        })

    def forward(self, images, input_ids, attention_mask, **kwargs):
        img_feat = self.dropout(self.image_encoder(images))
        txt_feat = self.dropout(
            self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            .last_hidden_state[:, 0]
        )
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return {task: self.heads[task](fused) for task in self.tasks}


class EarlyFusionTrainer:
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        save_path: str = "./models/early_fusion_model/model_weights.pt",
        batch_size: int = 16,
        epochs: int = 10,
        dropout: float = 0.2,
        lr_text: float = 2e-5,
        lr_image: float = 1e-4,
        lr_heads: float = 1e-4,
        weight_decay: float = 1e-2,
        mlflow_experiment: str = "EarlyFusion",
        use_mlflow: bool = True,
        results_path: str = "./results/early_fusion/training_results.csv",
    ) -> None:
        self.device          = "cuda" if torch.cuda.is_available() else "cpu"
        self.tasks           = list(EarlyFusionModel.DEFAULT_TASKS)
        self.save_path       = save_path
        self.images_dir      = images_dir
        self.dropout         = dropout
        self.weight_decay    = weight_decay
        self.mlflow_experiment = mlflow_experiment
        self.use_mlflow      = use_mlflow
        self.results_path    = results_path

        self.batch_size = batch_size
        self.epochs     = epochs
        self.lr_text    = lr_text
        self.lr_image   = lr_image
        self.lr_heads   = lr_heads

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.train_df, self.val_df, self.test_df, _ = _load_and_split_data(csv_path)

        self.model = EarlyFusionModel(dropout=dropout).to(self.device)
        self.criterions = self._build_criterions()

        self._rebuild_loaders_and_optimizer()

    def _build_loader(self, df, shuffle: bool) -> DataLoader:
        mean, std = compute_mean_std(self.images_dir, self.train_df["image_name"].tolist())
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return DataLoader(
            FusionDataset(df, self.images_dir, self.tokenizer, transform),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def _build_criterions(self) -> dict[str, nn.CrossEntropyLoss]:
        criterions = {}
        for task in self.tasks:
            weights = compute_class_weight(
                "balanced",
                classes=np.array([0, 1]),
                y=self.train_df[task].values,
            )
            w = torch.tensor(weights, dtype=torch.float).to(self.device)
            criterions[task] = nn.CrossEntropyLoss(weight=w)
        return criterions

    def _build_optimizer(self) -> AdamW:
        return AdamW([
            {"params": self.model.text_encoder.parameters(),  "lr": self.lr_text},
            {"params": self.model.image_encoder.parameters(), "lr": self.lr_image},
            {"params": self.model.heads.parameters(),         "lr": self.lr_heads},
        ], weight_decay=self.weight_decay)

    def _rebuild_loaders_and_optimizer(self) -> None:
        self.train_loader = self._build_loader(self.train_df, shuffle=True)
        self.val_loader   = self._build_loader(self.val_df,   shuffle=False)
        self.test_loader  = self._build_loader(self.test_df,  shuffle=False)
        self.optimizer    = self._build_optimizer()
        self.scheduler    = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    def _snapshot_params(self) -> dict:
        return {
            "batch_size":  self.batch_size,
            "epochs":      self.epochs,
            "lr_text":     self.lr_text,
            "lr_image":    self.lr_image,
            "lr_heads":    self.lr_heads,
        }

    def _apply_hyperparams(self, params: dict) -> None:
        self.batch_size = params.get("batch_size", self.batch_size)
        self.epochs     = params.get("epochs",     self.epochs)
        self.lr_text    = params.get("learning_rate",  self.lr_text)
        self.lr_image   = params.get("lr_finetune",    self.lr_image)
        self.lr_heads   = params.get("lr_finetune",    self.lr_heads)
        self.current_params = self._snapshot_params()
        self._rebuild_loaders_and_optimizer()

    def _unpack_batch(self, batch: dict) -> tuple[dict, dict]:
        inputs = {
            "images":         batch["image"].to(self.device),
            "input_ids":      batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
        }
        labels = {task: batch["labels"][task].to(self.device) for task in self.tasks}
        return inputs, labels

    def _compute_loss(self, logits: dict, labels: dict) -> torch.Tensor:
        return sum(self.criterions[task](logits[task], labels[task]) for task in self.tasks)

    def _run_train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            inputs, labels = self._unpack_batch(batch)
            logits = self.model(**inputs)
            loss   = self._compute_loss(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _run_eval_epoch(self, loader: DataLoader) -> tuple[float, dict, dict]:
        self.model.eval()
        y_true     = {task: [] for task in self.tasks}
        y_pred     = {task: [] for task in self.tasks}
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                inputs, labels = self._unpack_batch(batch)
                logits = self.model(**inputs)
                total_loss += self._compute_loss(logits, labels).item()

                for task in self.tasks:
                    preds = torch.argmax(logits[task], dim=1)
                    y_pred[task].extend(preds.cpu().numpy())
                    y_true[task].extend(labels[task].cpu().numpy())

        return total_loss / len(loader), y_true, y_pred

    def _train_single_with_results(self) -> tuple[float, list[dict], dict]:
        all_results  = []
        best_val_acc = 0.0
        best_epoch   = 0
        last_metrics = {}

        for epoch in range(1, self.epochs + 1):
            train_loss               = self._run_train_epoch()
            val_loss, y_true, y_pred = self._run_eval_epoch(self.val_loader)
            self.scheduler.step()

            per_task = {
                task: MultiTaskModel.compute_metrics(y_true[task], y_pred[task])
                for task in self.tasks
            }
            avg_acc = sum(m["acc"] for m in per_task.values()) / len(self.tasks)

            print(
                f"Epoch {epoch:02d}/{self.epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {avg_acc:.4f}"
            )

            for task, m in per_task.items():
                all_results.append({
                    "epoch":         epoch,
                    "task":          task,
                    "train_loss":    train_loss,
                    "val_loss":      val_loss,
                    "val_accuracy":  m["acc"],
                    "val_precision": m["precision"],
                    "val_recall":    m["recall"],
                    "val_f1":        m["f1"],
                    **self.current_params,
                })

            if avg_acc > best_val_acc:
                best_val_acc = avg_acc
                best_epoch   = epoch
                last_metrics = per_task
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  → Saved model (val_acc: {avg_acc:.4f})")

        print(f"\nBest epoch: {best_epoch} | val_acc: {best_val_acc:.4f}")
        return best_val_acc, all_results, last_metrics

    def _train_and_eval(self, params: dict) -> tuple[dict, float]:
        self._apply_hyperparams(params)
        val_acc, results, per_task = self._train_single_with_results()
        save_results_csv(results, self.results_path)
        return per_task, val_acc

    def train(self, hyperparams: list[dict] | None = None):
        self.current_params = self._snapshot_params()

        if hyperparams is not None:
            return self.model.grid_search(
                param_grid=hyperparams,
                eval_fn=self._train_and_eval,
                run_name_fn=lambda p: "_".join(f"{k}={v}" for k, v in p.items()),
                mlflow_experiment=self.mlflow_experiment,
                use_mlflow=self.use_mlflow,
                results_path=self.results_path,
            )

        val_acc, results, _ = self._train_single_with_results()
        save_results_csv(results, self.results_path)
        return val_acc

    def test(self) -> None:
        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device)
        )
        _, y_true, y_pred = self._run_eval_epoch(self.test_loader)
        for task in self.tasks:
            print_task_metrics(y_true, y_pred, task)