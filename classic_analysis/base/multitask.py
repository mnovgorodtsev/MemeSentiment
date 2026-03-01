import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from classic_analysis.datasets_preparation import _load_and_split_data, MemotionDataset, ImageMultiTaskDataset


class MultiTaskModel(nn.Module):
    def __init__(self, tasks=None):
        super().__init__()
        self.tasks = tasks or ["humour", "sarcasm", "offensive", "motivational"]

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented by subclasses!")
    

class MultiTaskTrainer:
    def __init__(self, model: MultiTaskModel, csv_path, data_type="text", images_dir=None,
                 save_path="./multitask_model", batch_size=32, epochs=3, learning_rate=2e-5,
                 lr_finetune=1e-4, freeze_epochs=0, max_length=128, num_workers=2, test=False):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
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

        self.train_df, self.test_df, self.encoders = _load_and_split_data(csv_path)

        if self.data_type == "text":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.tokenize()

        self.train_loader, self.test_loader = self.build_loaders()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if test:
            self.load_model()

    def tokenize(self):
        for feature in ["input_ids", "attention_mask"]:
            for df_set in [self.train_df, self.test_df]:
                df_set[feature] = list(df_set["text_corrected"].apply(
                    lambda x: self.tokenizer(x, padding="max_length", truncation=True, max_length=self.max_length)[feature]
                ))

    def build_loaders(self):
        if self.data_type == "text":
            train_dataset = MemotionDataset(self.train_df)
            test_dataset = MemotionDataset(self.test_df)
        else:  # image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            train_dataset = ImageMultiTaskDataset(self.train_df, self.images_dir, transform)
            test_dataset = ImageMultiTaskDataset(self.test_df, self.images_dir, transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, test_loader

    def train(self):
        for epoch in range(self.epochs):
            if self.data_type == "image" and epoch == self.freeze_epochs:
                for name, param in self.model.base.named_parameters():
                    param.requires_grad = True
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr_finetune)

            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()

                if self.data_type == "text":
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    images = batch[0].to(self.device)
                    labels = {task: batch[1][task].to(self.device) for task in self.model.tasks}
                    outputs = self.model(images)

                loss = sum(
                    self.criterion(outputs[task], batch[task].to(self.device) if self.data_type=="text" else labels[task])
                    for task in self.model.tasks
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(self.train_loader):.4f}")

        self.save_model()

    def save_model(self):
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_weights.pt"))

        if self.data_type == "text":    # encoders and tokenizer
            self.tokenizer.save_pretrained(self.save_path)
            with open(os.path.join(self.save_path, "label_encoders.pkl"), "wb") as f:
                pickle.dump(self.encoders, f)

        print(f"Model saved to {self.save_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, "model_weights.pt"), map_location=self.device))
        
        if self.data_type == "text":
            with open(os.path.join(self.save_path, "label_encoders.pkl"), "rb") as f:
                self.encoders = pickle.load(f)
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.save_path)

        self.model.eval()
        print(f"Loaded model from {self.save_path}")

    def test(self):
        self.model.eval()
        y_true = {task: [] for task in self.model.tasks}
        y_pred = {task: [] for task in self.model.tasks}

        with torch.no_grad():
            for batch in self.test_loader:
                if self.data_type == "text":
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                    labels = {task: batch[task].to(self.device) for task in self.model.tasks}
                else:
                    images = batch[0].to(self.device)
                    labels = {task: batch[task].to(self.device) for task in self.model.tasks}
                    outputs = self.model(images)

                for task in self.model.tasks:
                    preds = torch.argmax(outputs[task], dim=1)
                    y_true[task].extend(labels[task].cpu().numpy())
                    y_pred[task].extend(preds.cpu().numpy())

        for task in self.model.tasks:
            self.metrics(y_true, y_pred, task)

    @staticmethod
    def metrics(y_true, y_pred, task):
        print(f"\n===== {task.upper()} =====")
        acc = accuracy_score(y_true[task], y_pred[task])
        precision, recall, f1, _ = precision_recall_fscore_support(y_true[task], y_pred[task], average="weighted")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}\n")
        print(classification_report(y_true[task], y_pred[task], digits=4, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_true[task], y_pred[task]))