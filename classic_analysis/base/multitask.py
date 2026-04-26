import logging
import warnings

import os
import pickle

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer
import random
from sklearn.utils.class_weight import compute_class_weight

from classic_analysis.base.helpers import (
    compute_mean_std,
    print_task_metrics,
    save_results_csv,
)
from classic_analysis.datasets_preparation import (
    ImageMultiTaskDataset,
    TextMultiTaskDataset,
)

from utils.split_dataset import _load_and_split_data
import os

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MultiTaskModel(nn.Module):
    """
    Base class for multi-task models.

    Includes common logic for:
    - inference (run_inference)
    - evaluation (evaluate, print_evaluation)
    - grid search over arbitrary hyperparameters (grid_search)
    - MLflow logging (opt-out via use_mlflow=False)
    """

    DEFAULT_TASKS = ("humour", "sarcasm", "offensive", "motivational")

    def __init__(self, tasks: list[str] | None = None) -> None:
        super().__init__()
        self.tasks = list(tasks or self.DEFAULT_TASKS)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward().")

    def run_inference(self, loader: DataLoader, unpack_fn) -> tuple[dict, dict]:
        """
        Runs the model on a DataLoader and collects predictions.

        Args:
            loader:    DataLoader with the data.
            unpack_fn: function (batch, device) -> (inputs_dict, labels_dict).

        Returns:
            (y_true, y_pred) — dictionaries mapping task -> list of values.
        """
        device = next(self.parameters()).device
        y_true = {task: [] for task in self.tasks}
        y_pred = {task: [] for task in self.tasks}

        self.eval()
        with torch.no_grad():
            for batch in loader:
                inputs, labels = unpack_fn(batch, device)
                outputs = self(**inputs)
                for task in self.tasks:
                    preds = torch.argmax(outputs[task], dim=1)
                    y_pred[task].extend(preds.cpu().numpy())
                    y_true[task].extend(labels[task].cpu().numpy())

        return y_true, y_pred

    @staticmethod
    def compute_metrics(y_true: list, y_pred: list) -> dict:
        """Returns a dictionary with accuracy/precision/recall/f1 for a single task."""
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}

    def evaluate(self, loader: DataLoader, unpack_fn) -> dict[str, dict]:
        """
        Runs inference and returns per-task metrics.

        Returns:
            Dict mapping task -> {"acc", "precision", "recall", "f1"}.
        """
        y_true, y_pred = self.run_inference(loader, unpack_fn)
        return {
            task: self.compute_metrics(y_true[task], y_pred[task])
            for task in self.tasks
        }

    def print_evaluation(self, loader: DataLoader, unpack_fn, save_path: str = None) -> None:
        y_true, y_pred = self.run_inference(loader, unpack_fn)
        for task in self.tasks:
            print_task_metrics(y_true, y_pred, task, save_path=save_path)

    @staticmethod
    def _mlflow_log_run(
        run_name: str,
        params: dict,
        metrics: dict[str, dict],
        avg_acc: float,
    ) -> None:
        """Logs a single run to MLflow (params + per-task metrics + avg_acc)."""
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metric("avg_accuracy", avg_acc)
            for task, m in metrics.items():
                for metric_name, value in m.items():
                    mlflow.log_metric(f"{task}_{metric_name}", value)

    def grid_search(
        self,
        param_grid: list[dict],
        eval_fn,
        run_name_fn=None,
        mlflow_experiment: str = "GridSearch",
        use_mlflow: bool = True,
        results_path: str | None = None,
    ) -> tuple[dict, float]:
        """
        Generic grid search that works on any model.

        Args:
            param_grid:        List of dictionaries with parameters to test.
            eval_fn:           Function (params) -> (per_task_metrics, avg_acc).
                            Responsible for applying params and running evaluation.
            run_name_fn:       Optional function (params) -> str with the MLflow run name.
                            Default: repr(params).
            mlflow_experiment: Name of the experiment in MLflow.
            use_mlflow:        Whether to log to MLflow (default True).
            results_path:      Path to save CSV with results (optional).

        Returns:
            (best_params, best_avg_acc)
        """
        if use_mlflow:
            mlflow.set_experiment(mlflow_experiment)

        summary = []
        best_acc = -1.0
        best_params = None

        for params in param_grid:
            logger.info(f"Grid search - params: {params}")
            per_task, avg_acc = eval_fn(params)

            run_name = run_name_fn(params) if run_name_fn else str(params)
            logger.info(f"avg_accuracy={avg_acc:.4f} | {run_name}")

            if use_mlflow:
                self._mlflow_log_run(run_name, params, per_task, avg_acc)

            summary.append(
                {
                    **params,
                    "avg_accuracy": avg_acc,
                    **{
                        f"{t}_{k}": v for t, m in per_task.items() for k, v in m.items()
                    },
                }
            )

            if avg_acc > best_acc:
                best_acc, best_params = avg_acc, params

        logger.info(f"Best params: {best_params} | avg_acc: {best_acc:.4f}")

        if results_path:
            save_results_csv(summary, results_path)

        return best_params, best_acc


class MultiTaskTrainer:
    def __init__(
        self,
        model: MultiTaskModel,
        csv_path: str,
        data_type: str = "text",
        images_dir: str | None = None,
        save_path: str = "./multitask_model",
        batch_size: int = 32,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        lr_finetune: float = 1e-4,
        freeze_epochs: int = 0,
        max_length: int = 128,
        num_workers: int = 2,
        mlflow_experiment: str = "MultiTaskTrainer",
        use_mlflow: bool = True,
        test: bool = False,
        results_path="./results/training_results.csv",
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_type = data_type
        self.save_path = save_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_finetune = lr_finetune
        self.freeze_epochs = freeze_epochs
        self.max_length = max_length
        self.num_workers = num_workers
        self.images_dir = images_dir
        self.mlflow_experiment = mlflow_experiment
        self.use_mlflow = use_mlflow
        self.results_path = results_path
        self.best_val_acc = -1

        self.model_class = model.__class__
        self.model_init_kwargs = self._extract_model_init_params(model)

        self.model = model.to(self.device)
        self.data_type = data_type

        self.train_df, self.val_df, self.test_df, self.encoders = _load_and_split_data(
            csv_path
        )

        if self.data_type == "text":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self._tokenize_splits()

        self._generator = torch.Generator()
        self._generator.manual_seed(SEED)

        self.train_loader, self.val_loader, self.test_loader = self._build_loaders()
        self.class_weights = self._build_class_weights()
        self.criterion = nn.CrossEntropyLoss()
        self.current_params = self._snapshot_params()
        self._reset_optimizer()

        if test:
            self.load_model()

    def _build_class_weights(self):
        weights = {}
        for task in self.model.tasks:
            class_weights = compute_class_weight(
                "balanced",
                classes=np.unique(self.train_df[task]),
                y=self.train_df[task],
            )
            weights[task] = torch.tensor(
                class_weights, dtype=torch.float32, device=self.device
            )
        return weights

    def _extract_model_init_params(self, model: MultiTaskModel) -> dict:
        BERT_PARAMS = {
            "BertLinear": {"model_name": "bert-base-uncased", "dropout": 0.1},
            "BertMLP": {
                "model_name": "bert-base-uncased",
                "hidden_dim": 256,
                "dropout": 0.2,
            },
            "BertDeepMLP": {
                "model_name": "bert-base-uncased",
                "hidden_dim": 256,
                "dropout": 0.3,
            },
        }
        RESNET_PARAMS = {
            "ResNetLinear": {"dropout": 0.1},
            "ResNetAttention": {"dropout": 0.2},
            "ResNetAdaptivePooling": {"dropout": 0.2, "hidden_dim": 256},
        }
        class_name = model.__class__.__name__
        if class_name in BERT_PARAMS:
            params = BERT_PARAMS[class_name].copy()
        elif class_name in RESNET_PARAMS:
            params = RESNET_PARAMS[class_name].copy()
        else:
            params = {}
        params["tasks"] = model.tasks
        return params

    def _reinit_model(self) -> None:
        logger.info(f"Reinitializing model: {self.model_class.__name__}")
        self.model = self.model_class(**self.model_init_kwargs)
        self.model.to(self.device)
        self._reset_optimizer()

    def _tokenize_splits(self) -> None:
        for df in (self.train_df, self.val_df, self.test_df):
            for feature in ("input_ids", "attention_mask"):
                df[feature] = (
                    df["text_corrected"]
                    .apply(
                        lambda x: self.tokenizer(
                            x,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_length,
                        )[feature]
                    )
                    .tolist()
                )

    def _build_text_datasets(self) -> tuple:
        return (
            TextMultiTaskDataset(self.train_df),
            TextMultiTaskDataset(self.val_df),
            TextMultiTaskDataset(self.test_df),
        )

    def _build_image_datasets(self) -> tuple:
        mean, std = compute_mean_std(
            self.images_dir,
            self.train_df["image_name"].tolist(),
        )
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return (
            ImageMultiTaskDataset(self.train_df, self.images_dir, transform),
            ImageMultiTaskDataset(self.val_df, self.images_dir, transform),
            ImageMultiTaskDataset(self.test_df, self.images_dir, transform),
        )

    def _build_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_ds, val_ds, test_ds = (
            self._build_text_datasets()
            if self.data_type == "text"
            else self._build_image_datasets()
        )

        make_loader = lambda ds, shuffle: DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            generator=self._generator,
        )
        return (
            make_loader(train_ds, shuffle=True),
            make_loader(val_ds, shuffle=False),
            make_loader(test_ds, shuffle=False),
        )

    def _snapshot_params(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "lr_finetune": self.lr_finetune,
        }

    def _reset_optimizer(self, lr: float | None = None) -> None:
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr or self.learning_rate,
            weight_decay=1e-4,
        )

    def _unpack_batch(self, batch) -> tuple[dict, dict]:
        if self.data_type == "text":
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
            labels = {task: batch[task].to(self.device) for task in self.model.tasks}
        else:
            images, labels_dict = batch
            inputs = {"images": images.to(self.device)}
            labels = {
                task: labels_dict[task].to(self.device) for task in self.model.tasks
            }
        return inputs, labels

    def _forward(self, inputs: dict):
        if self.data_type == "text":
            return self.model(inputs["input_ids"], inputs["attention_mask"])
        return self.model(inputs["images"])

    def train(self, hyperparams: list[dict] | None = None):
        """
        Trains the model. If hyperparameters are provided, delegates to grid_search
        with MultiTaskModel.

        Returns:
            (best_params, best_val_acc) if hyperparameters are provided,
            otherwise best_val_acc.
        """
        if hyperparams is not None:
            return self.model.grid_search(
                param_grid=hyperparams,
                eval_fn=self._train_and_eval,
                run_name_fn=lambda p: "_".join(f"{k}={v}" for k, v in p.items()),
                mlflow_experiment=self.mlflow_experiment,
                use_mlflow=self.use_mlflow,
                results_path=self.results_path,
            )
        return self._train_single()

    def _train_and_eval(self, params: dict) -> tuple[dict, float]:
        self._apply_hyperparams(params)
        self._reinit_model()
        val_acc, results, per_task = self._train_single_with_results()
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        return per_task, val_acc

    def _apply_hyperparams(self, params: dict) -> None:
        self.batch_size = params.get("batch_size", self.batch_size)
        self.epochs = params.get("epochs", self.epochs)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.lr_finetune = params.get("lr_finetune", self.lr_finetune)
        self.current_params = self._snapshot_params()
        self.train_loader, self.val_loader, self.test_loader = self._build_loaders()

    def _train_single(self) -> float:
        val_acc, results, _ = self._train_single_with_results()
        save_results_csv(results, self.results_path)
        return val_acc

    def _train_single_with_results(self, patience: int = 2) -> tuple[float, list[dict], dict]:
        all_results = []
        best_val_acc = 0.0
        best_epoch = 0
        best_epoch_metrics = {}
        epoch_accs = []
        patience_counter = 0
    
        for epoch in range(self.epochs):
            self._maybe_unfreeze(epoch)
            avg_train_loss = self._run_train_epoch()
            val_metrics = self._run_eval_epoch(self.val_loader)
    
            epoch_results, avg_acc = self._log_and_collect(
                epoch, avg_train_loss, val_metrics
            )
            all_results.extend(epoch_results)
            epoch_accs.append(avg_acc)
    
            if self.use_mlflow:
                self._log_epoch_mlflow(epoch, avg_train_loss, val_metrics, avg_acc)
    
            if avg_acc > best_val_acc:
                best_val_acc = avg_acc
                best_epoch = epoch + 1
                best_epoch_metrics = val_metrics 
                patience_counter = 0  
                
                self.save_model()
                logger.info(f"Model improved. Saving checkpoint at epoch {best_epoch}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best epoch was {best_epoch} with acc: {best_val_acc:.4f}"
                    )
                    break
    
        avg_acc_all_epochs = sum(epoch_accs) / len(epoch_accs) if epoch_accs else 0.0
    
        logger.info(f"Best epoch: {best_epoch} | best val acc: {best_val_acc:.4f}")
        logger.info(f"Average accuracy across all epochs: {avg_acc_all_epochs:.4f}")
        logger.info(f"Total epochs trained: {len(epoch_accs)}")
    
        return best_val_acc, all_results, best_epoch_metrics

    def _log_epoch_mlflow(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, dict],
        avg_acc: float,
    ) -> None:
        run_name = "_".join(f"{k}={v}" for k, v in self.current_params.items())
        with mlflow.start_run(run_name=f"{run_name}_epoch{epoch+1}", nested=True):
            mlflow.log_params(self.current_params)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("avg_val_accuracy", avg_acc, step=epoch)
            for task, m in val_metrics.items():
                for metric_name, value in m.items():
                    mlflow.log_metric(f"{task}_{metric_name}", value, step=epoch)

    def _maybe_unfreeze(self, epoch: int) -> None:
        if self.data_type == "image" and epoch == self.freeze_epochs:
            for param in self.model.base.parameters():
                param.requires_grad = True
            self._reset_optimizer(lr=self.lr_finetune)

    def _run_train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            inputs, labels = self._unpack_batch(batch)
            outputs = self._forward(inputs)
            loss = 0
            for task in self.model.tasks:
                task_loss = nn.functional.cross_entropy(
                    outputs[task], labels[task], weight=self.class_weights[task]
                )
                loss += task_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _run_eval_epoch(self, loader: DataLoader) -> dict[str, dict]:
        self.model.eval()
        y_true = {task: [] for task in self.model.tasks}
        y_pred = {task: [] for task in self.model.tasks}

        with torch.no_grad():
            for batch in loader:
                inputs, labels = self._unpack_batch(batch)
                outputs = self._forward(inputs)
                for task in self.model.tasks:
                    preds = torch.argmax(outputs[task], dim=1)
                    y_true[task].extend(labels[task].cpu().numpy())
                    y_pred[task].extend(preds.cpu().numpy())

        return {
            task: MultiTaskModel.compute_metrics(y_true[task], y_pred[task])
            for task in self.model.tasks
        }

    def _log_and_collect(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, dict],
    ) -> tuple[list[dict], float]:
        rows, acc_list = [], []
        for task, m in val_metrics.items():
            logger.info(
                f"Epoch {epoch+1} - {task.upper()} | "
                f"Val Acc: {m['acc']:.4f}, Precision: {m['precision']:.4f}, "
                f"Recall: {m['recall']:.4f}, F1: {m['f1']:.4f}"
            )
            acc_list.append(m["acc"])
            rows.append(
                {
                    "epoch": epoch + 1,
                    "task": task,
                    "train_loss": train_loss,
                    "val_accuracy": m["acc"],
                    "val_precision": m["precision"],
                    "val_recall": m["recall"],
                    "val_f1": m["f1"],
                    **self.current_params,
                }
            )
        return rows, sum(acc_list) / len(acc_list)

    def test(self, txt_path: str) -> None:
        self.model.print_evaluation(self.test_loader, self._unpack_batch_for_model, txt_path)

    def _unpack_batch_for_model(self, batch, device) -> tuple[dict, dict]:
        """Adapter unpack_fn compatible with MultiTaskModel.run_inference."""
        inputs, labels = self._unpack_batch(batch)
        return inputs, labels

    def save_model(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_path, "model_weights.pt"),
        )
        if self.data_type == "text":
            self.tokenizer.save_pretrained(self.save_path)
            with open(os.path.join(self.save_path, "label_encoders.pkl"), "wb") as f:
                pickle.dump(self.encoders, f)
        logger.info(f"Model saved to {self.save_path}")

    def load_model(self) -> None:
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.save_path, "model_weights.pt"),
                map_location=self.device,
            )
        )
        if self.data_type == "text":
            with open(os.path.join(self.save_path, "label_encoders.pkl"), "rb") as f:
                self.encoders = pickle.load(f)
            self.tokenizer = AutoTokenizer.from_pretrained(self.save_path)
        self.model.eval()
        logger.info(f"Model loaded from {self.save_path}")


def initialize_model(model, 
                     csv_path: str = "data/memotion_dataset_7k/labels.csv",
                     save_path: str = "./models/bert_multitask", 
                     results_path: str = "./results/bert/training_results.csv",
                     images_dir: str = "data/memotion_dataset_7k/images",
                     data_type: str = "text",
                     test: bool = False):
    trainer = MultiTaskTrainer(
        model=model,
        csv_path=csv_path,
        data_type=data_type,
        save_path=save_path,
        results_path=results_path,
        images_dir=images_dir,
        use_mlflow=False,
        test=test
    )
    return trainer