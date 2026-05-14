import logging
import os
import json
import base64
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from vllm_analysis.prompts import (
    get_tasks,
    get_prompt_for_task,
    get_few_shot_examples,
    parse_classification_response,
)

logger = logging.getLogger(__name__)

load_dotenv()


class MemeClassifierNonOpenSource:
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
    ):

        self.model_name = model_name

        self.client = OpenAI(
            api_key=os.getenv("API_KEY")
        )

        logger.info(
            f"Using OpenAI model: {self.model_name}"
        )

    def _encode_image(
        self,
        image_path: str,
    ) -> str:

        with open(image_path, "rb") as f:
            return base64.b64encode(
                f.read()
            ).decode("utf-8")

    def _generate_image_caption(
        self,
        image_path: str,
        image_data: str,
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

            return response.output_text.strip()

        except Exception as e:

            logger.error(
                f"Caption generation failed for "
                f"{image_path}: {e}"
            )

            return "caption unavailable"

    def _classify_for_task(
        self,
        task: str,
        meme_text: str,
        image_caption: str,
        image_data: str,
        train_df: pd.DataFrame = None,
    ) -> str:

        try:
            few_shot_context = get_few_shot_examples(
                train_df,
                task,
                n_shots=2,
            )

            inputs = {
                "meme_text": meme_text,
                "image_description": image_caption,
                "few_shot_examples": few_shot_context,
            }

            prompt = get_prompt_for_task(
                task,
                **inputs,
            )

            print(prompt)

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

            logger.error(
                f"Classification error "
                f"for task '{task}': {e}"
            )

            return "error"

    def classify_meme(
        self,
        row: pd.Series,
        image_path: str,
        train_df: pd.DataFrame = None,
    ) -> Dict[str, str]:

        tasks = get_tasks()

        results = {}

        meme_text = str(
            row.get("text_corrected", "")
        ).strip()

        image_data = self._encode_image(
            image_path
        )

        image_caption = (
            self._generate_image_caption(
                image_path,
                image_data,
            )
        )

        for task in tasks:

            results[task] = (
                self._classify_for_task(
                    task=task,
                    meme_text=meme_text,
                    image_caption=image_caption,
                    image_data=image_data,
                    train_df=train_df,
                )
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

            image_filename = (
                row.get("image_name")
                or row.get("image")
            )

            if pd.isna(image_filename):
                continue

            image_path = os.path.join(
                images_base_path,
                str(image_filename),
            )

            if not os.path.exists(image_path):

                logger.warning(
                    f"Image not found: "
                    f"{image_path}"
                )

                continue

            logger.info(
                f"Processing sample {idx}"
            )

            responses = self.classify_meme(
                row,
                image_path,
                train_df,
            )

            for task in tasks:

                raw_response = (
                    responses.get(
                        task,
                        "error",
                    )
                )

                prediction = (
                    parse_classification_response(
                        raw_response,
                        task,
                    )
                )

                batch_results[task][
                    "predictions"
                ].append(prediction)

                batch_results[task][
                    "true_labels"
                ].append(int(row[task]))

                batch_results[task][
                    "raw_responses"
                ].append(raw_response)

                batch_results[task][
                    "sample_ids"
                ].append(idx)

        return batch_results
    