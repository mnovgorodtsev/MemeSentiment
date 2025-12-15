import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset


class DataPreparation:
    def __init__(self, path="../data/memotion_dataset_7k/labels.csv", column_to_classify="humour"):
        self.df = pd.read_csv(path)
        self.columns_to_drop = ["Unnamed: 0", "text_ocr", "overall_sentiment"]
        self.column_to_classify = column_to_classify
        self.label_column = "text_corrected"
        self.encoder = LabelEncoder()

    def join_labels(self):
        self.df["humour"] = self.df["humour"].replace(
            ["hilarious", "very_funny", "funny"],
            "funny"
        )
        self.df["sarcasm"] = self.df["sarcasm"].replace(
            ["twisted_meaning", "very_twisted"],
            "twisted"
        )
        self.df["offensive"] = self.df["offensive"].replace(
            ["very_offensive", "slight", "hateful_offensive"],
            "offensive"
        )

    def remove_nans(self):
        self.df.dropna(inplace=True)

    def drop_columns(self):
        self.df = self.df.drop(self.columns_to_drop, axis=1)

    def convert_to_string(self):
        self.df[self.label_column] = self.df[self.label_column].astype(str)
        self.df[self.column_to_classify] = self.df[self.column_to_classify].astype(str)

    def label_encode(self):
        self.df[self.column_to_classify] = self.encoder.fit_transform(
            self.df[self.column_to_classify]
        )

    def split_to_test_and_train(self):
        dataset = Dataset.from_pandas(
            self.df[[self.label_column, self.column_to_classify]]
        )
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        return dataset["train"], dataset["test"]

    def prepare_data(self):
        self.remove_nans()
        self.drop_columns()
        self.join_labels()
        self.convert_to_string()
        self.label_encode()
        return self.df


data_preparation = DataPreparation()
