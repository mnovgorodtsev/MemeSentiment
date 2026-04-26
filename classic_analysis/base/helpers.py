import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm


def save_results_csv(
    results: list[dict], path: str = "./results/resnet/training_results.csv"
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"Results saved to {path}")


def compute_mean_std(
    images_dir: str,
    image_list: list[str],
    cache_path: str = "models/mean_std/mean_std.pt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean and std for a list of images, with disk caching."""
    if os.path.exists(cache_path):
        data = torch.load(cache_path)
        return data["mean"], data["std"]

    means, stds = [], []
    for img_name in tqdm(image_list, desc="Computing mean/std"):
        img_path = os.path.join(images_dir, img_name)
        try:
            img = np.array(Image.open(img_path).convert("RGB")) / 255.0
        except Exception:
            continue
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))

    mean = torch.tensor(np.mean(means, axis=0), dtype=torch.float32)
    std = torch.tensor(np.mean(stds, axis=0), dtype=torch.float32)

    torch.save({"mean": mean, "std": std}, cache_path)
    print(f"Mean/std cached to {cache_path}.")
    return mean, std

 
def print_task_metrics(y_true: dict, y_pred: dict, task: str, save_path: str = None) -> None:
    output_lines = []
    output_lines.append(f"\n===== {task.upper()} =====")
    
    acc = accuracy_score(y_true[task], y_pred[task])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true[task], y_pred[task], average="weighted", zero_division=0
    )
    
    output_lines.append(f"Accuracy : {acc:.4f}")
    output_lines.append(f"Precision: {precision:.4f}")
    output_lines.append(f"Recall   : {recall:.4f}")
    output_lines.append(f"F1-score : {f1:.4f}\n")
    
    clf_report = classification_report(y_true[task], y_pred[task], digits=4, zero_division=0)
    output_lines.append(clf_report)
    
    output_lines.append("Confusion matrix:")
    output_lines.append(str(confusion_matrix(y_true[task], y_pred[task])))
    
    full_output = "\n".join(output_lines)
    print(full_output)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a") as f:
            f.write(full_output + "\n")
