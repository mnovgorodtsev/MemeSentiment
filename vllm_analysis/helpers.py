import json
import logging
from typing import Dict, List
from vllm_analysis.prompts import get_tasks
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid_mask = y_pred != -1

    if not valid_mask.any():
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "valid_predictions": 0,
            "total_predictions": len(y_pred),
            "error_rate": 1.0,
            "confusion_matrix": {
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_positives": 0,
            },
        }

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
    tn, fp = cm[0]
    fn, tp = cm[1]

    return {
        "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
        "precision": float(
            precision_score(y_true_valid, y_pred_valid, zero_division=0)
        ),
        "recall": float(recall_score(y_true_valid, y_pred_valid, zero_division=0)),
        "f1": float(f1_score(y_true_valid, y_pred_valid, zero_division=0)),
        "valid_predictions": int(valid_mask.sum()),
        "total_predictions": len(y_pred),
        "error_rate": float((~valid_mask).sum() / len(y_pred)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def format_task_metrics(task: str, metrics: Dict) -> str:
    lines = [
        f"  {task.upper():^40}",
        f"    Accuracy:           {metrics['accuracy']:.4f}",
        f"    Precision:          {metrics['precision']:.4f}",
        f"    Recall:             {metrics['recall']:.4f}",
        f"    F1-Score:           {metrics['f1']:.4f}",
        f"    Valid Predictions:  {metrics['valid_predictions']}/{metrics['total_predictions']}",
        f"    Error Rate:         {metrics['error_rate']:.4f}",
    ]

    cm = metrics.get("confusion_matrix", {})
    if cm:
        lines.append(
            f"    Confusion Matrix:   TP={cm['true_positives']}, TN={cm['true_negatives']}, "
            f"FP={cm['false_positives']}, FN={cm['false_negatives']}"
        )

    return "\n".join(lines)


def generate_report(all_results: List[Dict]) -> str:
    report = []
    report.append("=" * 100)
    report.append("MEME CLASSIFICATION BENCHMARK REPORT")
    report.append("=" * 100)

    for result in all_results:
        if result is None:
            continue

        model = result["model"]
        dataset_size = result["dataset_size"]
        duration = result["duration_seconds"]

        report.append(f"\n{'MODEL':<30} {model}")
        report.append(f"{'Dataset Size':<30} {dataset_size}")
        report.append(f"{'Total Duration':<30} {duration:.2f} seconds")
        report.append("-" * 100)

        for task, task_data in result["tasks"].items():
            metrics = task_data["metrics"]
            report.append(format_task_metrics(task, metrics))

        report.append("")

    report.append("\n" + "=" * 100)
    report.append("SUMMARY - CROSS-MODEL COMPARISON")
    report.append("=" * 100)

    tasks = get_tasks()

    for task in tasks:
        report.append(f"\n{task.upper()}:")
        report.append(
            f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} "
            f"{'Recall':<12} {'F1-Score':<12} {'Duration':<12}"
        )
        report.append("-" * 80)

        for result in all_results:
            if result is None or task not in result["tasks"]:
                continue

            metrics = result["tasks"][task]["metrics"]
            model = result["model"]
            duration = result["duration_seconds"]

            report.append(
                f"{model:<20} {metrics['accuracy']:<12.4f} "
                f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                f"{metrics['f1']:<12.4f} {duration:<12.2f}s"
            )

    return "\n".join(report)


def save_report(report: str, filepath: str) -> None:
    with open(filepath, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {filepath}")


def save_detailed_results(all_results: List[Dict], filepath: str) -> None:
    json_results = []

    for result in all_results:
        if result is None:
            continue

        clean_result = result.copy()
        for task in clean_result.get("tasks", {}):
            clean_result["tasks"][task]["predictions"] = list(
                clean_result["tasks"][task]["predictions"]
            )
            clean_result["tasks"][task]["true_labels"] = list(
                clean_result["tasks"][task]["true_labels"]
            )
            clean_result["tasks"][task]["sample_ids"] = list(
                clean_result["tasks"][task]["sample_ids"]
            )

        json_results.append(clean_result)

    with open(filepath, "w") as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"Detailed results saved to {filepath}")
