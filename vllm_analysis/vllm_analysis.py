import logging
import time
import numpy as np
from typing import Dict

from utils.read_config import Config
from vllm_analysis.classifier import MemeClassifier
from vllm_analysis.prompts import get_tasks
from vllm_analysis.helpers import (
    calculate_metrics,
    generate_report,
    save_report,
    save_detailed_results,
)
from utils.split_dataset import _load_and_split_data
from vllm_analysis.non_open_source import MemeClassifierNonOpenSource

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_model_on_dataset(
    model_name: str,
    test_df,
    images_base_path: str,
    train_df,
    max_samples: int = None,
    use_few_shot: bool = True,
    classifier_class=None,
    classifier_kwargs=None,
) -> Dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing model: {model_name}")
    logger.info(f"Max samples: {max_samples or 'all'}")
    logger.info(f"Few-shot prompting: {use_few_shot}")
    logger.info(f"{'='*60}")

    test_subset = test_df.iloc[:max_samples] if max_samples else test_df
    test_size = len(test_subset)

    if classifier_class is None:
        config = Config()
        classifier_class = MemeClassifier
        classifier_kwargs = {"host": config.ollama_host, "model_name": model_name}

    try:
        classifier = classifier_class(**classifier_kwargs)
    except RuntimeError as e:
        logger.error(str(e))
        return None

    start_time = time.time()
    batch_results = classifier.classify_batch(
        test_subset,
        images_base_path,
        train_df=train_df,
        use_few_shot=use_few_shot,
    )
    end_time = time.time()

    results = {
        "model": model_name,
        "dataset_size": test_size,
        "use_few_shot": use_few_shot,
        "tasks": {},
        "timestamps": {
            "start": start_time,
            "end": end_time,
        },
        "duration_seconds": end_time - start_time,
    }

    tasks = get_tasks()
    
    for task in tasks:
        y_true = np.array(batch_results[task]["true_labels"])
        y_pred = np.array(batch_results[task]["predictions"])
        metrics = calculate_metrics(y_true, y_pred)
        
        results["tasks"][task] = {
            "predictions": batch_results[task]["predictions"],
            "true_labels": batch_results[task]["true_labels"],
            "raw_responses": batch_results[task]["raw_responses"],
            "sample_ids": batch_results[task]["sample_ids"],
            "metrics": metrics,
        }
        
        logger.info(
            f"{task.upper()}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"Valid={metrics['valid_predictions']}/{metrics['total_predictions']}"
        )
    
    return results


def main():
    logger.info("Starting meme classification benchmark...")

    config = Config()
    logger.info(f"Configuration: {config.to_dict()}")

    logger.info(f"Loading dataset from {config.memotion_dataset_path}")
    train_df, val_df, test_df, _ = _load_and_split_data(config.memotion_dataset_path)
    logger.info(f"Test set size: {len(test_df)}")

    all_results = []

    result = test_model_on_dataset(
        model_name=config.ollama_model,
        test_df=test_df,
        images_base_path=config.images_base_path,
        train_df=train_df,
        max_samples=3,
        use_few_shot=True,
    )
    all_results.append(result)

    report = generate_report(all_results)
    print(report)

    save_report(report, "meme_classification_report.txt")
    save_detailed_results(all_results, "meme_classification_results.json")

    logger.info("Benchmark completed!")


def main_gpt():
    logger.info("Starting meme classification benchmark (GPT-4)...")

    config = Config()
    logger.info(f"Configuration: {config.to_dict()}")

    logger.info(f"Loading dataset from {config.memotion_dataset_path}")
    train_df, val_df, test_df, _ = _load_and_split_data(config.memotion_dataset_path)
    logger.info(f"Test set size: {len(test_df)}")

    all_results = []

    result = test_model_on_dataset(
        model_name="gpt-4.1-mini",
        test_df=test_df,
        images_base_path=config.images_base_path,
        train_df=train_df,
        max_samples=10,
        use_few_shot=True,
        classifier_class=MemeClassifierNonOpenSource,
        classifier_kwargs={"model_name": "gpt-4.1-mini"},
    )
    all_results.append(result)

    report = generate_report(all_results)
    print(report)

    save_report(report, "meme_classification_report_gpt-4.1-mini.txt")
    save_detailed_results(all_results, "meme_classification_results_gpt-4.1-mini.json")

    logger.info("Benchmark completed!")


if __name__ == "__main__":
    main_gpt()