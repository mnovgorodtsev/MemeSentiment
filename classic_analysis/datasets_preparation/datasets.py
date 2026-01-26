import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _load_and_split_data(csv_path, test_size=0.1, random_state=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text_corrected", "humour", "image_name"])

    df["humour"] = df["humour"].replace(
        ["hilarious", "very_funny", "funny"], "funny"
    )

    df["text_corrected"] = df["text_corrected"].astype(str)
    df["humour"] = df["humour"].astype(str)

    encoder = LabelEncoder()
    df["humour"] = encoder.fit_transform(df["humour"])

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["humour"]
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), encoder


class BertHumourDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long)
        label = torch.tensor(row["humour"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }


class ImageHumourDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.images_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["humour"])
        return image, label


class FusionDataset(Dataset):
    def __init__(self, df, images_dir, tokenizer, transform, max_length=128):
        self.df = df
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

        label = int(row["humour"])

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label)
        }
