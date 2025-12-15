from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from data_preparation import data_preparation


class BertHumourTrainer:
    def __init__(
        self,
        model_name="bert-base-uncased",
        output_dir="./bert_humour",
        save_path="./bert_humour_model",
        max_length=128,
        batch_size=8,
        epochs=1,
        learning_rate=2e-5,
        weight_decay=0.01
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.save_path = save_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.trainer = None

    def tokenize(self, batch):
        return self.tokenizer(
            batch[data_preparation.label_column],
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

    def prepare_datasets(self):
        data_preparation.prepare_data()
        train, test = data_preparation.split_to_test_and_train()

        train_ds = train.map(self.tokenize, batched=True)
        test_ds = test.map(self.tokenize, batched=True)

        train_ds = train_ds.rename_column(
            data_preparation.column_to_classify, "labels"
        )
        test_ds = test_ds.rename_column(
            data_preparation.column_to_classify, "labels"
        )

        train_ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        test_ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        return train_ds, test_ds

    def build_model(self):
        num_labels = len(data_preparation.encoder.classes_)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )

    def train(self):
        train_ds, test_ds = self.prepare_datasets()
        self.build_model()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            logging_steps=50,
            report_to="none"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds
        )

        self.trainer.train()
        self.trainer.evaluate()

    def save(self):
        if self.trainer is None:
            raise RuntimeError("Model has not been trained yet!")

        self.trainer.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)


trainer = BertHumourTrainer(
    epochs=1,
    batch_size=8
)

trainer.train()
trainer.save()
