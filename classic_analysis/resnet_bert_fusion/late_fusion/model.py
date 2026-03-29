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
from classic_analysis.datasets_preparation import FusionDataset, _load_and_split_data


def fusion_unpack(batch: dict, device: str) -> tuple[dict, dict]:
    inputs = {
        "image": batch["image"].to(device),
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    labels = {task: batch["labels"][task] for task in batch["labels"]}
    return inputs, labels


class ResnetWrapper(MultiTaskModel):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        super().__init__()
        base = resnet18()
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.heads = nn.ModuleDict(
            {task: nn.Linear(in_features, 2) for task in self.tasks}
        )
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.to(device)
        self.eval()

    def forward(self, images, **kwargs):
        return {task: self.heads[task](self.base(images)) for task in self.tasks}


class BertWrapper(MultiTaskModel):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, 2) for task in self.tasks})
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.to(device)
        self.eval()

    def forward(self, input_ids, attention_mask, **kwargs):
        cls = self.dropout(
            self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0]
        )
        return {task: self.heads[task](cls) for task in self.tasks}


class LateFusionModel(MultiTaskModel):
    def __init__(
        self,
        resnet_path: str,
        bert_path: str,
        w_image: float = 0.5,
        w_text: float = 0.5,
    ) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._resnet_path = resnet_path
        self._bert_path = bert_path
        self.w_image = w_image
        self.w_text = w_text
        self.image_model = ResnetWrapper(resnet_path, self.device)
        self.text_model = BertWrapper(bert_path, self.device)

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
                    "resnet_path": self._resnet_path,
                    "bert_path": self._bert_path,
                    "w_image": self.w_image,
                    "w_text": self.w_text,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "LateFusionModel":
        with open(path) as f:
            return cls(**json.load(f))


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
        self.resnet_path = resnet_path
        self.bert_path = bert_path
        self.batch_size = batch_size
        self.save_path = save_path
        self.results_path = results_path
        self.mlflow_experiment = mlflow_experiment
        self.use_mlflow = use_mlflow

        self.train_df, self.val_df, self.test_df, _ = _load_and_split_data(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
        return LateFusionModel(self.resnet_path, self.bert_path, w_image, w_text)

    def _eval_fn(self, params: dict) -> tuple[dict, float]:
        model = self._build_model(params["w_image"], params["w_text"])
        per_task = model.evaluate(self.val_loader, fusion_unpack)
        avg_acc = sum(m["acc"] for m in per_task.values()) / len(model.tasks)
        return per_task, avg_acc

    def train(self, hyperparams: list[dict] | None = None) -> tuple[dict, float]:
        """Search for best weights on validation dataset and save to JSON."""
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

    def test(self) -> None:
        """Load best weights on test dataset."""
        try:
            model = LateFusionModel.load(self.save_path)
            model.print_evaluation(self.test_loader, fusion_unpack)
        except Exception as e:
            raise FileNotFoundError(f"First train the weights! {e}")
