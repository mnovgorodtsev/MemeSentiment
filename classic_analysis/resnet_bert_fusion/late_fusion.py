import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from classic_analysis.datasets_preparation import _load_and_split_data, FusionDataset


class ResnetWrapper(nn.Module):
    def __init__(self, model_path, device="cpu"):
        super().__init__()
        self.device = device
        self.model = resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x):
        return self.model(x)


class BertWrapper(nn.Module):
    def __init__(self, model_path, device="cpu"):
        super().__init__()
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class LateFusionModel(nn.Module):
    def __init__(self, resnet_path, bert_path, device="cpu", w_image=0.5, w_text=0.5):
        super().__init__()
        self.device = device
        self.image_model = ResnetWrapper(resnet_path, device=device)
        self.text_model = BertWrapper(bert_path, device=device)
        self.w_image = w_image
        self.w_text = w_text

    def forward(self, image, input_ids, attention_mask):
        logits_img = self.image_model(image)
        logits_txt = self.text_model(input_ids, attention_mask)
        return self.w_image * logits_img + self.w_text * logits_txt


class LateFusionEvaluator:
    def __init__(self, csv_path, images_dir, resnet_path, bert_path, batch_size=16, test_size=0.1, random_state=42):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        _, test_df, _ = _load_and_split_data(csv_path, test_size=test_size, random_state=random_state)

        self.dataset = FusionDataset(test_df, images_dir, self.tokenizer, self.transform)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        self.model = LateFusionModel(resnet_path, bert_path, device=self.device).to(self.device)

    def evaluate(self):
        y_true, y_pred = [], []

        self.model.eval()
        with torch.no_grad():
            for batch in self.loader:
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(images, input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        print("Accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluator = LateFusionEvaluator(
        csv_path="../../data/memotion_dataset_7k/new_labels.csv",
        images_dir="../../data/memotion_dataset_7k/images",
        resnet_path="../resnet_pipeline/resnet_model/resnet_model.pth",
        bert_path="../bert_pipeline/bert_humour_model",
        batch_size=16
    )

    evaluator.evaluate()
