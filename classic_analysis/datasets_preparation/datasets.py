import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _load_and_split_data(csv_path):

    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["text_corrected"])
    df["text_corrected"] = df["text_corrected"].astype(str)

    df = _binarize_labels(df)

    tasks = ["humour", "sarcasm", "offensive", "motivational"]

    encoders = {}

    for task in tasks:
        enc = LabelEncoder()
        df[task] = enc.fit_transform(df[task].astype(str))
        encoders[task] = enc

    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df["humour"]
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), encoders


def _binarize_labels(df):
    # humour
    df["humour"] = df["humour"].replace({
        "funny": "funny",
        "hilarious": "funny",
        "very_funny": "funny",
        "not_funny": "not_funny"
    })

    # sarcasm
    df["sarcasm"] = df["sarcasm"].replace({
        "general": "not_sarcastic",
        "not_sarcastic": "not_sarcastic",
        "twisted_meaning": "sarcastic",
        "very_twisted": "sarcastic"
    })

    # offensive
    df["offensive"] = df["offensive"].replace({
        "not_offensive": "not_offensive",
        "slight": "offensive",
        "very_offensive": "offensive",
        "hateful_offensive": "offensive"
    })

    return df


class MemotionDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),

            "humour": torch.tensor(row["humour"], dtype=torch.long),
            "sarcasm": torch.tensor(row["sarcasm"], dtype=torch.long),
            "offensive": torch.tensor(row["offensive"], dtype=torch.long),
            "motivational": torch.tensor(row["motivational"], dtype=torch.long)
        }


class ImageMultiTaskDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.tasks = ["humour", "sarcasm", "offensive", "motivational"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.images_dir, row["image_name"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.df))

        if self.transform:
            image = self.transform(image)

        labels = {
            "humour": int(row["humour"]),
            "sarcasm": int(row["sarcasm"]),
            "offensive": int(row["offensive"]),
            "motivational": int(row["motivational"])
        }

        return image, labels


class FusionDataset(Dataset):

    def __init__(self, df, images_dir, tokenizer, transform, max_length=128):

        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img_path = os.path.join(self.images_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        encoding = self.tokenizer(
            row["text_corrected"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = {
            "humour": int(row["humour"]),
            "sarcasm": int(row["sarcasm"]),
            "offensive": int(row["offensive"]),
            "motivational": int(row["motivational"])
        }

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }
