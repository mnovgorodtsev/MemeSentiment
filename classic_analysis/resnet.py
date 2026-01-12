import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MemeHumourDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.images_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["humour"])  # 0 albo 1
        return image, label


train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

from torch.utils.data import DataLoader

dataset = MemeHumourDataset(
    csv_path="../data/memotion_dataset_7k/new_labels.csv",
    images_dir="../data/memotion_dataset_7k/images",
    transform=train_tfms
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 0/1
model = model.to(device)


for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)


FREEZE_EPOCHS = 3

for epoch in range(10):
    if epoch == FREEZE_EPOCHS:
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )
        print(">>> Unfroze layer4")

    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss={total_loss:.4f}")


model.eval()
img = Image.open("../data/memotion_dataset_7k/images/image_23.jpeg").convert("RGB")
x = train_tfms(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1).item()

print("funny" if pred == 1 else "not funny")
