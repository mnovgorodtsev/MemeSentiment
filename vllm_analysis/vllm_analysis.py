import logging
import time
import argparse
import numpy as np
from typing import Dict, Optional

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


def load_data(config: Config, english_dataset: bool) -> tuple:
    results_dir = config.results_dir

    if english_dataset:
        logger.info(f"Loading English dataset from {config.memotion_dataset_path}")
        train_df, _, test_df, _ = _load_and_split_data(config.memotion_dataset_path)
        base_path = config.images_base_path
        txt_path = f"{results_dir}/test_meme_classification_report_{{model_type}}.txt"
        json_path = f"{results_dir}/test_meme_classification_results_{{model_type}}.json"
        logger.info(f"Test set size: {len(test_df)}")
    else:
        logger.info(f"Loading Polish dataset from {config.polish_dataset_path}")
        train_df, test_df = _load_and_split_data_polish(config.polish_dataset_path)
        base_path = config.polish_base_path
        txt_path = f"{results_dir}/test_meme_classification_report_{{model_type}}_polish.txt"
        json_path = f"{results_dir}/test_meme_classification_results_{{model_type}}_polish.json"

    return train_df, test_df, base_path, txt_path, json_path


def run_benchmark(
    model_name: str,
    test_df,
    images_base_path: str,
    train_df,
    max_samples: Optional[int] = None,
    use_few_shot: bool = False,
    model_type: str = "ollama",
    english_dataset: bool = True,
    **kwargs,
) -> Optional[Dict]:
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing model: {model_name}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Max samples: {max_samples or 'all'}")
    logger.info(f"Few-shot prompting: {use_few_shot}")
    logger.info(f"{'='*60}")

    test_subset = test_df.iloc[:max_samples] if max_samples else test_df

    try:
        classifier = create_classifier(model_type=model_type, model_name=model_name, **kwargs)
    except RuntimeError as e:
        logger.error(str(e))
        return None

    start_time = time.time()
    batch_results = classifier.classify_batch(
        test_subset,
        images_base_path,
        train_df=train_df,
        use_few_shot=use_few_shot,
        english_dataset=english_dataset,
    )
    duration = time.time() - start_time

    return _build_results(
        model_name=model_name,
        batch_results=batch_results,
        test_size=len(test_subset),
        use_few_shot=use_few_shot,
        start_time=start_time,
        duration=duration,
        english_dataset=english_dataset,
    )


def _build_results(
    model_name: str,
    batch_results: Dict,
    test_size: int,
    use_few_shot: bool,
    start_time: float,
    duration: float,
    english_dataset: bool,
) -> Dict:
    results = {
        "model": model_name,
        "dataset_size": test_size,
        "use_few_shot": use_few_shot,
        "duration_seconds": duration,
        "timestamps": {"start": start_time, "end": start_time + duration},
        "tasks": {},
    }

    for task in get_tasks(english_dataset):
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

    model_cfg = config.model_configs.get(model_type)
    if model_cfg is None:
        raise ValueError(f"Unknown model_type: '{model_type}'. Must be one of: {list(config.model_configs)}")

    train_df, test_df, base_path, txt_path, json_path = load_data(config, english_dataset)
    txt_path = txt_path.format(model_type=model_type)
    json_path = json_path.format(model_type=model_type)

    logger.info(f"Starting meme classification benchmark - {model_cfg['log_name']}")

    result = run_benchmark(
        model_name=model_cfg["model_name"],
        test_df=test_df,
        images_base_path=base_path,
        train_df=train_df,
        max_samples=max_samples,
        model_type=model_type,
        english_dataset=english_dataset,
        **model_cfg["extra_kwargs"],
    )

    if result is None:
        logger.error("Failed to run benchmark")
        return

    report = generate_report([result])
    save_report(report, txt_path)
    save_detailed_results([result], json_path)


if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser(description="Meme Classification Benchmark")
    parser.add_argument("--model_type", choices=list(config.model_configs), default="ollama")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--dataset", choices=["en", "pl"], default="en")
    args = parser.parse_args()
    main(model_type=args.model_type, max_samples=args.max_samples, english_dataset=False if args.dataset == 'pl' else True)