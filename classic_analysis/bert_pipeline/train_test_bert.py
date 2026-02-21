import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from classic_analysis.datasets_preparation import _load_and_split_data, MemotionDataset
import os

import torch
import torch.nn as nn
from transformers import AutoModel
import pickle


class MultiTaskBert(nn.Module):

    def __init__(self, model_name, num_labels_dict):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.humour_head = nn.Linear(hidden, num_labels_dict["humour"])
        self.sarcasm_head = nn.Linear(hidden, num_labels_dict["sarcasm"])
        self.offensive_head = nn.Linear(hidden, num_labels_dict["offensive"])
        self.motivational_head = nn.Linear(hidden, num_labels_dict["motivational"])

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)

        return {
            "humour": self.humour_head(pooled),
            "sarcasm": self.sarcasm_head(pooled),
            "offensive": self.offensive_head(pooled),
            "motivational": self.motivational_head(pooled),
        }


class BertMultiTaskTrainer:

    def __init__(
        self,
        csv_path,
        model_name="bert-base-uncased",
        save_path="./bert_multitask_model",
        max_length=128,
        batch_size=32,
        epochs=3,
        learning_rate=2e-5,
        test=False
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.save_path = save_path

        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.train_df, self.test_df, self.encoders = _load_and_split_data(csv_path)

        self.tokenize()

        self.train_loader = DataLoader(
            MemotionDataset(self.train_df),
            batch_size=batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            MemotionDataset(self.test_df),
            batch_size=batch_size
        )

        num_labels_dict = {
            task: len(self.encoders[task].classes_)
            for task in self.encoders
        }

        self.model = MultiTaskBert(model_name, num_labels_dict).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.criterion = nn.CrossEntropyLoss()

        if test:
            self.load_model()

    def tokenize(self):

        for feature in ["input_ids", "attention_mask"]:

            for df_set in [self.train_df, self.test_df]:

                df_set[feature] = list(
                    df_set["text_corrected"].apply(
                        lambda x: self.tokenizer(
                            x,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_length
                        )[feature]
                    )
                )

    def train(self):

        self.model.train()

        for epoch in range(self.epochs):

            total_loss = 0

            for batch in self.train_loader:

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask)

                loss_h = self.criterion(outputs["humour"], batch["humour"].to(self.device))
                loss_s = self.criterion(outputs["sarcasm"], batch["sarcasm"].to(self.device))
                loss_o = self.criterion(outputs["offensive"], batch["offensive"].to(self.device))
                loss_m = self.criterion(outputs["motivational"], batch["motivational"].to(self.device))

                loss = loss_h + loss_s + loss_o + loss_m

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} Loss: {total_loss/len(self.train_loader):.4f}")

        self.save_model()


    def save_model(self):

        os.makedirs(self.save_path, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "encoders": self.encoders
        }, os.path.join(self.save_path, "model.pt"))

        self.tokenizer.save_pretrained(self.save_path)

        with open(os.path.join(self.save_path, "label_encoders.pkl"), "wb") as f:
            pickle.dump(self.encoders, f)

        print(f"Model saved to {self.save_path}")

    def load_model(self):

        checkpoint = torch.load(
            os.path.join(self.save_path, "model.pt"),
            map_location=self.device,
            weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.encoders = checkpoint["encoders"]

        self.model.eval()

        print(f"Loaded model from {self.save_path}")


    def test(self):

        self.model.eval()

        tasks = ["humour", "sarcasm", "offensive", "motivational"]

        y_true = {task: [] for task in tasks}
        y_pred = {task: [] for task in tasks}

        with torch.no_grad():

            for batch in self.test_loader:

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask)

                for task in tasks:

                    labels = batch[task].to(self.device)

                    preds = torch.argmax(outputs[task], dim=1)

                    y_true[task].extend(labels.cpu().numpy())
                    y_pred[task].extend(preds.cpu().numpy())

        for task in tasks:

            print(f"\n===== {task.upper()} =====")

            self.metrics(
                y_true[task],
                y_pred[task],
                self.encoders[task]
            )


    def metrics(self, y_true, y_pred, encoder):

        acc = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted"
        )

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}\n")

        print(classification_report(
            y_true,
            y_pred,
            target_names=encoder.classes_,
            digits=4,
            zero_division=0
        ))

        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":

    trainer = BertMultiTaskTrainer(
        csv_path="data/memotion_dataset_7k/labels.csv",
        epochs=20,
        batch_size=32,
        test=True
    )

    # trainer.train()
    trainer.test()