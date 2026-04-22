import logging
import time
import numpy as np
from typing import Dict

from config import Config
from vllm_analysis.classifier import MemeClassifier
from vllm_analysis.helpers import (
    calculate_metrics,
    generate_report,
    save_report,
    save_detailed_results,
)
from utils.split_dataset import _load_and_split_data

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
    use_few_shot: bool = False,
) -> Dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing model: {model_name}")
    logger.info(f"Max samples: {max_samples or 'all'}")
    logger.info(f"Few-shot prompting: {use_few_shot}")
    logger.info(f"{'='*60}")

    test_subset = test_df.iloc[:max_samples] if max_samples else test_df
    test_size = len(test_subset)

    config = Config()
    try:
        classifier = MemeClassifier(config.ollama_host, model_name)
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

    from prompts import TASKS

    for task in TASKS:
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

    models_to_test = [config.ollama_model]
    all_results = []

    for model in models_to_test:
        result = test_model_on_dataset(
            model_name=model,
            test_df=test_df,
            images_base_path=config.images_base_path,
            train_df=train_df,
            max_samples=3,
            use_few_shot=False,
        )
        all_results.append(result)

    report = generate_report(all_results)
    print(report)

    save_report(report, "meme_classification_report_2204.txt")
    save_detailed_results(all_results, "meme_classification_results_2204.json")

    logger.info("Benchmark completed!")


if __name__ == "__main__":
    main()
