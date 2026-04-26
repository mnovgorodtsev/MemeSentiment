import logging
import os
from typing import Dict
import pandas as pd
import ollama

from vllm_analysis.prompts import get_tasks, get_prompt_for_task, get_few_shot_examples, parse_classification_response
from utils.read_config import Config

logger = logging.getLogger(__name__)


class MemeClassifier:
    def __init__(self, ollama_host: str = None, model_name: str = None):
        config = Config()
        self.client = ollama.Client(host=ollama_host or config.ollama_host)
        self.model_name = model_name or config.ollama_model
        self._verify_model_available()

    def _verify_model_available(self) -> None:
        try:
            self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
            )
            logger.info(f"Model '{self.model_name}' is available")
        except Exception as e:
            raise RuntimeError(
                f"Model '{self.model_name}' not available or Ollama not accessible: {e}"
            )

    def classify_meme(
        self,
        row: pd.Series,
        image_path: str,
        train_df: pd.DataFrame = None,
        use_few_shot: bool = False,
    ) -> Dict[str, str]:
        tasks = get_tasks()
        results = {}
        meme_text = str(row.get("text_corrected", "")).strip()

        for task in tasks:
            try:
                prompt = get_prompt_for_task(task, meme_text)

                if use_few_shot and train_df is not None:
                    few_shot_context = get_few_shot_examples(
                        train_df, task, n_shots=2
                    )
                    prompt = f"{few_shot_context}\n\n{prompt}"

                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_path],
                        }
                    ],
                    stream=False,
                )
                results[task] = response["message"]["content"].strip().lower()

            except Exception as e:
                logger.error(f"Error classifying {image_path} for task '{task}': {e}")
                results[task] = "error"

        return results

    def classify_batch(
        self,
        df: pd.DataFrame,
        images_base_path: str,
        train_df: pd.DataFrame = None,
        use_few_shot: bool = False,
    ) -> Dict[str, list]:
        tasks = get_tasks()

        batch_results = {
            task: {
                "predictions": [],
                "true_labels": [],
                "raw_responses": [],
                "sample_ids": [],
            }
            for task in tasks
        }

        for idx, row in df.iterrows():
            image_filename = row.get("image_name") or row.get("image")
            if pd.isna(image_filename):
                logger.warning(f"No image name for row {idx}")
                continue

            image_path = os.path.join(images_base_path, str(image_filename))

            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue

            responses = self.classify_meme(row, image_path, train_df, use_few_shot)

            for task in tasks:
                raw_response = responses.get(task, "error")
                prediction = parse_classification_response(
                    raw_response, task
                )

                batch_results[task]["predictions"].append(prediction)
                batch_results[task]["true_labels"].append(int(row[task]))
                batch_results[task]["raw_responses"].append(raw_response)
                batch_results[task]["sample_ids"].append(idx)

        return batch_results
