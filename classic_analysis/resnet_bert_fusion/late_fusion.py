import logging

import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from torchvision.models import resnet18
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import os
import mlflow
from classic_analysis.datasets_preparation import _load_and_split_data, FusionDataset
from classic_analysis.base import MultiTaskModel
from logging import getLogger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = getLogger(__name__)

def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ResnetWrapper(MultiTaskModel):

    def __init__(self, model_path, device="cpu"):
        super().__init__()

        self.device = device

        base = resnet18()
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base

        self.heads = nn.ModuleDict({
            task: nn.Linear(in_features, 2)
            for task in self.tasks
        })

        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)

        self.to(device)
        self.eval()

    def forward(self, images):
        features = self.base(images)
        return {task: self.heads[task](features) for task in self.tasks}


class BertWrapper(MultiTaskModel):

    def __init__(self, model_path, device="cpu"):
        super().__init__()

        self.device = device

        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden, 2)
            for task in self.tasks
        })

        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)

        self.to(device)
        self.eval()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0])
        return {task: self.heads[task](cls) for task in self.tasks}


class LateFusionModel(MultiTaskModel):

    def __init__(self, resnet_path, bert_path, w_image=0.5, w_text=0.5):
        super().__init__()
        self.device = _get_device()
        self.image_model = ResnetWrapper(resnet_path, self.device)
        self.text_model = BertWrapper(bert_path, self.device)
        self.w_image = w_image
        self.w_text = w_text

    def forward(self, image, input_ids, attention_mask):
        img_logits = self.image_model(image)
        txt_logits = self.text_model(input_ids, attention_mask)

        return {
            task: self.w_image * img_logits[task] + self.w_text * txt_logits[task]
            for task in self.tasks
        }


class LateFusionEvaluator:
    mlflow.set_tracking_uri("./mlruns")
    WEIGHT_GRID = [(round(w/10, 1), round(1 - w/10, 1)) for w in range(1, 10)]

    def __init__(
        self,
        csv_path,
        images_dir,
        resnet_path,
        bert_path,
        batch_size=16,
        results_dir="./results/late_fusion",
        mlflow_experiment="LateFusion_GridSearch",
        run_name="default"
    ):
        self.device = _get_device()
        self.resnet_path = resnet_path
        self.bert_path = bert_path
        self.results_dir = results_dir
        self.tasks = ["humour", "sarcasm", "offensive", "motivational"]

        os.makedirs(results_dir, exist_ok=True)
        mlflow.set_experiment(mlflow_experiment)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        _, test_df, _ = _load_and_split_data(csv_path)

        self.loader = DataLoader(
            FusionDataset(test_df, images_dir, self.tokenizer, self.transform),
            batch_size=batch_size,
            shuffle=False
        )

        mlflow.set_experiment(mlflow_experiment)

    def _run_inference(self, model):

        y_true = {task: [] for task in self.tasks}
        y_pred = {task: [] for task in self.tasks}

        model.eval()

        with torch.no_grad():
            for batch in self.loader:
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                logits = model(images, input_ids, attention_mask)

                for task in self.tasks:
                    preds = torch.argmax(logits[task], dim=1)
                    y_pred[task].extend(preds.cpu().numpy())
                    y_true[task].extend(labels[task].cpu().numpy())

        return y_true, y_pred

    def grid_search(self):
        logger.info(f"Running grid search... {self.WEIGHT_GRID}")
        summary = []
        best_acc = -1
        best_weights = None

        for w_image, w_text in self.WEIGHT_GRID:

            logger.info(f"Testing: w_image={w_image:.1f}, w_text={w_text:.1f}")

            model = LateFusionModel(
                self.resnet_path,
                self.bert_path,
                w_image=w_image,
                w_text=w_text
            )

            y_true, y_pred = self._run_inference(model)

            per_task_acc = {
                task: accuracy_score(y_true[task], y_pred[task])
                for task in self.tasks
            }
            avg_acc = sum(per_task_acc.values()) / len(self.tasks)

            with mlflow.start_run(run_name=f"w_img={w_image}_w_txt={w_text}"):
                mlflow.log_param("w_image", w_image)
                mlflow.log_param("w_text", w_text)
                mlflow.log_metric("avg_accuracy", avg_acc)
                for task, acc in per_task_acc.items():
                    mlflow.log_metric(f"acc_{task}", acc)

            logger.info(f"avg_accuracy={avg_acc:.4f} | {per_task_acc}")

            summary.append({
                "w_image": w_image,
                "w_text": w_text,
                "avg_accuracy": avg_acc,
                **{f"acc_{t}": per_task_acc[t] for t in self.tasks}
            })

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_weights = (w_image, w_text)

        summary_path = os.path.join(self.results_dir, "grid_search_summary.csv")
        pd.DataFrame(summary).sort_values("avg_accuracy", ascending=False).to_csv(
            summary_path, index=False
        )

        logger.info(f"Best wages: w_image={best_weights[0]}, w_text={best_weights[1]}")
        logger.info(f"avg_accuracy: {best_acc:.4f}")
        logger.info(f"Saved results to: {summary_path}")

        return best_weights

    def evaluate_best(self, w_image, w_text):

        model = LateFusionModel(
            self.resnet_path,
            self.bert_path,
            w_image=w_image,
            w_text=w_text
        )

        y_true, y_pred = self._run_inference(model)

        for task in self.tasks:
            print(f"\n===== {task.upper()} =====")
            print("Accuracy:", accuracy_score(y_true[task], y_pred[task]))
            print(classification_report(y_true[task], y_pred[task], digits=4, zero_division=0))
            print("Confusion matrix:")
            print(confusion_matrix(y_true[task], y_pred[task]))

if __name__ == "__main__":
    evaluator = LateFusionEvaluator(
        csv_path="data/memotion_dataset_7k/labels.csv",
        images_dir="data/memotion_dataset_7k/images",
        resnet_path="./models/resnet_multitask_model/model_weights.pt",
        bert_path="./models/bert_multitask_model/model_weights.pt",
        batch_size=32,
        run_name="default_run"
    )

    best_w_image, best_w_text = evaluator.grid_search()
    evaluator.evaluate_best(best_w_image, best_w_text)