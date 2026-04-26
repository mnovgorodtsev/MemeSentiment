import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from transformers import AutoModel, AutoTokenizer

from classic_analysis.base import MultiTaskModel
from classic_analysis.base.helpers import compute_mean_std
from classic_analysis.datasets_preparation import FusionDataset
from utils.split_dataset import _load_and_split_data
from classic_analysis.bert_pipeline.model import BertDeepMLP
from classic_analysis.resnet_pipeline.model import ResNetLinear


def fusion_unpack(batch: dict, device: str) -> tuple[dict, dict]:
    inputs = {
        "image": batch["image"].to(device),
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    labels = {task: batch["labels"][task] for task in batch["labels"]}
    return inputs, labels


class LateFusionModel(MultiTaskModel):
    def __init__(
        self,
        image_model: MultiTaskModel,
        text_model: MultiTaskModel,
        w_image: float = 0.5,
        w_text: float = 0.5,
    ) -> None:
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.w_image = w_image
        self.w_text = w_text

    def forward(self, image, input_ids, attention_mask, **kwargs):
        img_logits = self.image_model(image)
        txt_logits = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return {
            task: self.w_image * img_logits[task] + self.w_text * txt_logits[task]
            for task in self.tasks
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "w_image": self.w_image,
                    "w_text": self.w_text,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(
        cls, path: str, image_model: MultiTaskModel, text_model: MultiTaskModel
    ) -> "LateFusionModel":
        with open(path) as f:
            params = json.load(f)
            return cls(image_model, text_model, **params)


class LateFusionTrainer:
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        resnet_path: str,
        bert_path: str,
        batch_size: int = 32,
        save_path: str = "./models/late_fusion_model/weights.json",
        results_path: str = "./results/late_fusion/grid_search_summary.csv",
        mlflow_experiment: str = "LateFusion_GridSearch",
        use_mlflow: bool = True,
    ) -> None:
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.save_path = save_path
        self.results_path = results_path
        self.mlflow_experiment = mlflow_experiment
        self.use_mlflow = use_mlflow
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_df, self.val_df, self.test_df, _ = _load_and_split_data(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.image_model = ResNetLinear()
        self.text_model = BertDeepMLP()

        self.image_model.load_state_dict(
            torch.load(resnet_path, map_location=self.device)
        )
        self.text_model.load_state_dict(torch.load(bert_path, map_location=self.device))

        self.image_model.to(self.device)
        self.text_model.to(self.device)
        self.image_model.eval()
        self.text_model.eval()

        self.val_loader = self._build_loader(self.val_df)
        self.test_loader = self._build_loader(self.test_df)

    def _build_loader(self, df) -> DataLoader:
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
        return DataLoader(
            FusionDataset(df, self.images_dir, self.tokenizer, transform),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def _build_model(
        self, w_image: float = 0.5, w_text: float = 0.5
    ) -> LateFusionModel:
        return LateFusionModel(self.image_model, self.text_model, w_image, w_text)

    def _eval_fn(self, params: dict) -> tuple[dict, float]:
        model = self._build_model(params["w_image"], params["w_text"])
        per_task = model.evaluate(self.val_loader, fusion_unpack)
        avg_acc = sum(m["acc"] for m in per_task.values()) / len(model.tasks)
        return per_task, avg_acc

    def train(self, hyperparams: list[dict] | None = None) -> tuple[dict, float]:
        proxy = self._build_model()
        best_params, best_acc = proxy.grid_search(
            param_grid=hyperparams,
            eval_fn=self._eval_fn,
            run_name_fn=lambda p: f"w_img={p['w_image']}_w_txt={p['w_text']}",
            mlflow_experiment=self.mlflow_experiment,
            use_mlflow=self.use_mlflow,
            results_path=self.results_path,
        )
        best_model = self._build_model(best_params["w_image"], best_params["w_text"])
        best_model.save(self.save_path)
        return best_params, best_acc

    def test(self, save_path: str = None) -> None:
        try:
            model = LateFusionModel.load(
                self.save_path, self.image_model, self.text_model
            )
            model.print_evaluation(self.test_loader, fusion_unpack, save_path=save_path)
        except Exception as e:
            raise FileNotFoundError(f"First train the weights! {e}")
