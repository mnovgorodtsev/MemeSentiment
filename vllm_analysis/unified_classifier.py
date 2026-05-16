import logging
import os
import base64
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from vllm_analysis.prompts import (
    get_tasks,
    get_prompt_for_task,
    get_few_shot_examples_with_images,
    get_few_shot_examples,
    parse_classification_response,
)

from utils.read_config import Config

logger = logging.getLogger(__name__)
load_dotenv()
config = Config()


class BaseClassifier(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        pass

    @abstractmethod
    def _generate_image_caption(
        self, image_path: str, image_data: str
    ) -> str:
        pass

    @abstractmethod
    def _classify_for_task(
        self,
        task: str,
        meme_text: str,
        image_caption: str,
        image_data: str,
        train_df: pd.DataFrame = None,
        use_few_shot: bool = False,
    ) -> str:
        pass

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

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

        image_data = self._encode_image(image_path)
        image_caption = self._generate_image_caption(image_path, image_data)

        for task in tasks:
            results[task] = self._classify_for_task(
                task=task,
                meme_text=meme_text,
                image_caption=image_caption,
                image_data=image_data,
                train_df=train_df,
                use_few_shot=use_few_shot,
            )

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

            logger.info(f"Processing sample {idx}")

            responses = self.classify_meme(
                row, image_path, train_df, use_few_shot
            )

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


class OllamaClassifier(BaseClassifier):

    def __init__(self, ollama_host: str = None, model_name: str = None):
        import ollama

        self.ollama = ollama
        self.ollama_host = ollama_host
        super().__init__(model_name or "llama2")

    def _initialize_client(self) -> None:
        self.client = self.ollama.Client(host=self.ollama_host)
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

    def _generate_image_caption(
        self, image_path: str, image_data: str
    ) -> str:
        try:
            caption_response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "Describe this image in detail. Focus on objects, people, text, and meme context.",
                        "images": [image_data],
                    }
                ],
                stream=False,
            )
            return caption_response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return "caption unavailable"

    def _classify_for_task(
        self,
        task: str,
        meme_text: str,
        image_caption: str,
        image_data: str,
        train_df: pd.DataFrame = None,
        use_few_shot: bool = False,
    ) -> str:
        try:
            few_shot_list = []
            if use_few_shot and train_df is not None:
                few_shot_list = get_few_shot_examples(
                    train_df, task, n_shots=2
                )

            inputs = {
                "meme_text": meme_text,
                "image_description": image_caption,
                "few_shot_examples": few_shot_list,
            }

            prompt = get_prompt_for_task(task, **inputs)

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_data],
                    }
                ],
                stream=False,
            )
            return response["message"]["content"].strip().lower()

        except Exception as e:
            logger.error(f"Error classifying for task '{task}': {e}")
            return "error"


class OpenAIClassifier(BaseClassifier):

    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(model_name)

    def _initialize_client(self) -> None:
        self.client = OpenAI(api_key=os.getenv("API_KEY"))
        logger.info(f"Using OpenAI model: {self.model_name}")

    def _generate_image_caption(
        self, image_path: str, image_data: str
    ) -> str:
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Describe this meme image in detail. "
                                    "Focus on people, objects, emotions, "
                                    "scene, meme context and visible text."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": (
                                    f"data:image/jpeg;base64,{image_data}"
                                ),
                            },
                        ],
                    }
                ],
                max_output_tokens=300,
            )

            return (
                response.output_text
                .strip()
                .lower()
            )

        except Exception as e:
            logger.error(
                f"Caption generation failed for {image_path}: {e}"
            )
            return "caption unavailable"

    def _classify_for_task(
        self,
        task: str,
        meme_text: str,
        image_caption: str,
        image_data: str,
        train_df: pd.DataFrame = None,
        use_few_shot: bool = False,
    ) -> str:
        try:
            few_shot_list = []
            if use_few_shot and train_df is not None:
                few_shot_list = get_few_shot_examples(
                    train_df, task, n_shots=2
                )

            inputs = {
                "meme_text": meme_text,
                "image_description": image_caption,
                "few_shot_examples": few_shot_list,
            }

            prompt = get_prompt_for_task(task, **inputs)

            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                            {
                                "type": "input_image",
                                "image_url": (
                                    f"data:image/jpeg;base64,{image_data}"
                                ),
                            },
                        ],
                    }
                ],
                max_output_tokens=100,
            )

            return (
                response.output_text
                .strip()
                .lower()
            )

        except Exception as e:
            logger.error(f"Classification error for task '{task}': {e}")
            return "error"


def create_classifier(
    model_type: str = "ollama",
    model_name: str = None,
    **kwargs,
) -> BaseClassifier:
    """
    Factory function to create the appropriate classifier.

    Args:
        model_type: Either "ollama" or "openai"
        model_name: Name of the model to use
        **kwargs: Additional arguments passed to the classifier

    Returns:
        An instance of the appropriate classifier
    """
    if model_type.lower() == "ollama":
        return OllamaClassifier(
            ollama_host=kwargs.get("ollama_host"),
            model_name=model_name,
        )
    elif model_type.lower() == "openai":
        return OpenAIClassifier(model_name=model_name or "gpt-4o-mini")
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Must be 'ollama' or 'openai'."
        )
