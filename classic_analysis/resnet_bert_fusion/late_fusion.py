import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from classic_analysis.datasets_preparation import _load_and_split_data, FusionDataset


class ResnetWrapper(nn.Module):

    def __init__(self, model_path, device="cpu"):
        super().__init__()

        self.device = device

        base = resnet18()
        in_features = base.fc.in_features
        base.fc = nn.Identity()

        self.base = base

        self.heads = nn.ModuleDict({
            "humour": nn.Linear(in_features, 2),
            "sarcasm": nn.Linear(in_features, 2),
            "offensive": nn.Linear(in_features, 2),
            "motivational": nn.Linear(in_features, 2)
        })

        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)

        self.to(device)
        self.eval()

    def forward(self, x):

        features = self.base(x)

        logits = {
            task: head(features)
            for task, head in self.heads.items()
        }

        return logits


class BertWrapper(nn.Module):

    def __init__(self, model_path, device="cpu"):
        super().__init__()

        self.device = device

        self.base = AutoModel.from_pretrained("bert-base-uncased")

        hidden = self.base.config.hidden_size

        self.heads = nn.ModuleDict({
            "humour": nn.Linear(hidden, 2),
            "sarcasm": nn.Linear(hidden, 2),
            "offensive": nn.Linear(hidden, 2),
            "motivational": nn.Linear(hidden, 2)
        })

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        new_state_dict = self.rename_heads(checkpoint["model_state_dict"])

        self.load_state_dict(new_state_dict)

        self.to(device)
        self.eval()

    def rename_heads(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("bert."):
                k = k.replace("bert.", "base.")
            if k.startswith("humour_head"):
                k = k.replace("humour_head", "heads.humour")
            if k.startswith("sarcasm_head"):
                k = k.replace("sarcasm_head", "heads.sarcasm")
            if k.startswith("offensive_head"):
                k = k.replace("offensive_head", "heads.offensive")
            if k.startswith("motivational_head"):
                k = k.replace("motivational_head", "heads.motivational")
            new_state_dict[k] = v
        return new_state_dict

    def forward(self, input_ids, attention_mask):

        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]

        logits = {
            task: head(cls)
            for task, head in self.heads.items()
        }

        return logits


class LateFusionModel(nn.Module):

    def __init__(self, resnet_path, bert_path, device="cpu", w_image=0.5, w_text=0.5):
        super().__init__()

        self.device = device

        self.image_model = ResnetWrapper(resnet_path, device)
        self.text_model = BertWrapper(bert_path, device)

        self.w_image = w_image
        self.w_text = w_text

        self.tasks = ["humour", "sarcasm", "offensive", "motivational"]

    def forward(self, image, input_ids, attention_mask):

        img_logits = self.image_model(image)
        txt_logits = self.text_model(input_ids, attention_mask)

        fused = {}

        for task in img_logits.keys():
            fused[task] = (
                self.w_image * img_logits[task] +
                self.w_text * txt_logits[task]
            )

        return fused
    

class LateFusionEvaluator:

    def __init__(
        self,
        csv_path,
        images_dir,
        resnet_path,
        bert_path,
        batch_size=16,
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tasks = [
            "humour",
            "sarcasm",
            "offensive",
            "motivational"
        ]

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        _, test_df, _ = _load_and_split_data(
            csv_path,
        )

        self.dataset = FusionDataset(
            test_df,
            images_dir,
            self.tokenizer,
            self.transform
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.model = LateFusionModel(
            resnet_path,
            bert_path,
            device=self.device
        ).to(self.device)

    def evaluate(self):

        y_true = {task: [] for task in self.tasks}
        y_pred = {task: [] for task in self.tasks}

        self.model.eval()

        with torch.no_grad():

            for batch in self.loader:

                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                labels = batch["labels"]

                logits = self.model(
                    images,
                    input_ids,
                    attention_mask
                )

                for task in self.tasks:

                    preds = torch.argmax(logits[task], dim=1)

                    y_pred[task].extend(
                        preds.cpu().numpy()
                    )

                    y_true[task].extend(
                        labels[task].cpu().numpy()
                    )

        for task in self.tasks:

            print(f"\n===== {task.upper()} =====")

            acc = accuracy_score(
                y_true[task],
                y_pred[task]
            )

            print("Accuracy:", acc)

            print(
                classification_report(
                    y_true[task],
                    y_pred[task],
                    digits=4,
                    zero_division=0
                )
            )

            print("Confusion matrix:")
            print(
                confusion_matrix(
                    y_true[task],
                    y_pred[task]
                )
            )


if __name__ == "__main__":
    evaluator = LateFusionEvaluator(
        csv_path="data/memotion_dataset_7k/labels.csv",
        images_dir="data/memotion_dataset_7k/images",
        resnet_path="models/resnet_multitask_model/model.pth",
        bert_path="models/bert_multitask_model/model.pt",
        batch_size=16
    )

    evaluator.evaluate()
