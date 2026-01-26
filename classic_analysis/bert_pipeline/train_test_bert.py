import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from classic_analysis.datasets_preparation import _load_and_split_data, BertHumourDataset


class BertHumourTorchTrainer:
    def __init__(
        self,
        csv_path,
        model_name="bert-base-uncased",
        save_path="./bert_humour_model",
        max_length=128,
        batch_size=8,
        epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_workers=2,
        test=False
    ):
        self.csv_path = csv_path
        self.model_name = model_name
        self.save_path = save_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.train_df, self.test_df, self.encoder = _load_and_split_data(csv_path)

        self.train_loader = DataLoader(
            BertHumourDataset(self.train_df, self.tokenizer, max_length),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_loader = DataLoader(
            BertHumourDataset(self.test_df, self.tokenizer, max_length),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.encoder.classes_)
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

        if test:
            self.load_model()


    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0

            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")

        self.save_model()


    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path + ".pt")
        self.tokenizer.save_pretrained(self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.save_path + ".pt", map_location=self.device))
        self.model.eval()
        print(f"Loaded model from {self.save_path}")


    def test(self):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        self.metrics(y_true, y_pred)


    def metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        print(f"\nAccuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}\n")

        print(classification_report(
            y_true,
            y_pred,
            target_names=self.encoder.classes_,
            digits=4,
            zero_division=0
        ))

        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    trainer = BertHumourTorchTrainer(
        csv_path="../../data/memotion_dataset_7k/labels.csv",
        epochs=4,
        batch_size=8
    )

    trainer.train()
    # trainer.test()
