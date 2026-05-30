import logging
import time
import argparse
import numpy as np
from typing import Dict

from utils.read_config import Config
from vllm_analysis.unified_classifier import create_classifier
from vllm_analysis.prompts import get_tasks
from vllm_analysis.helpers import (
    calculate_metrics,
    generate_report,
    save_report,
    save_detailed_results,
)
from utils.split_dataset import _load_and_split_data, _load_and_split_data_polish

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
    model_type: str = "ollama",
    english_dataset: bool = True,
    **kwargs,
) -> Dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing model: {model_name}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Max samples: {max_samples or 'all'}")
    logger.info(f"Few-shot prompting: {use_few_shot}")
    logger.info(f"{'='*60}")

    test_subset = test_df.iloc[:max_samples] if max_samples else test_df
    test_size = len(test_subset)

    try:
        classifier = create_classifier(
            model_type=model_type,
            model_name=model_name,
            **kwargs,
        )
    except RuntimeError as e:
        logger.error(str(e))
        return None

    start_time = time.time()
    batch_results = classifier.classify_batch(
        test_subset,
        images_base_path,
        train_df=train_df,
        use_few_shot=use_few_shot,
        english_dataset=english_dataset
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

    tasks = get_tasks(english_dataset)
    
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


def main(model_type: str = "ollama", max_samples: int = None, english_dataset: bool = True):
    config = Config()

    if english_dataset:
        logger.info(f"Loading dataset from {config.memotion_dataset_path}")
        train_df, _, test_df, _ = _load_and_split_data(config.memotion_dataset_path)
        base_path = config.images_base_path
        logger.info(f"Test set size: {len(test_df)}")
    else:
        # we don't use test df due to small amount of data
        logger.info(f"Loading dataset from {config.polish_dataset_path}")
        train_df, test_df = _load_and_split_data_polish(config.polish_dataset_path)
        base_path = config.polish_base_path

    if model_type == "ollama":
        logger.info("Starting meme classification benchmark - Ollama")
        model_name = config.ollama_model
        extra_kwargs = {"ollama_host": config.ollama_host}
    elif model_type == "openai":
        logger.info("Starting meme classification benchmark - OpenAI")
        model_name = "gpt-4o-mini"
        extra_kwargs = {}
    elif model_type == "llamacpp":
        logger.info("Starting meme classification benchmark - LlamaCpp")
        model_name = "MemeLens-VLM"
        extra_kwargs = {}
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'ollama' or 'openai'")

    all_results = []

    result = test_model_on_dataset(
        model_name=model_name,
        test_df=test_df,
        images_base_path=base_path,
        train_df=train_df,
        max_samples=max_samples,
        use_few_shot=False,
        model_type=model_type,
        english_dataset=english_dataset,
        **extra_kwargs,
    )

    if result is None:
        logger.error("Failed to run benchmark")
        return

    all_results.append(result)

    report = generate_report(all_results)

    save_report(report, "meme_classification_report_gemma3.txt")
    save_detailed_results(all_results, "meme_classification_results_gemma3.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meme Classification Benchmark"
    )
    parser.add_argument(
        "--model_type",
        choices=["ollama", "openai", "llamacpp"],
        default="ollama",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--english_dataset",
        type=bool,
        default=True,
    )

    args = parser.parse_args()

    main(model_type=args.model_type, max_samples=args.max_samples, english_dataset=args.english_dataset)