import os

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TextMultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, df, tasks=None):
        self.df = df
        self.tasks = tasks or ["humour", "sarcasm", "offensive", "motivational"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }
        labels = {
            task: torch.tensor(row[task], dtype=torch.long) for task in self.tasks
        }
        return {**inputs, **labels}


class ImageMultiTaskDataset(Dataset):
    def __init__(self, df, images_dir, transform=None, tasks=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform
        self.tasks = tasks or ["humour", "sarcasm", "offensive", "motivational"]

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

        labels = {task: int(row[task]) for task in self.tasks}

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
            return_tensors="pt",
        )

        labels = {
            "humour": int(row["humour"]),
            "sarcasm": int(row["sarcasm"]),
            "offensive": int(row["offensive"]),
            "motivational": int(row["motivational"]),
        }

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }
