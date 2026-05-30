import logging
import os
import base64
from typing import Dict, List
import pandas as pd
from utils.read_config import Config

logger = logging.getLogger(__name__)


def get_classification_config() -> Dict:
    config = Config()
    return config.classification_config


def get_tasks(use_english_dataset: bool = True) -> list:
    config = get_classification_config()
    return list(config.get("tasks", {}).keys()) if use_english_dataset else list(config.get("polish_tasks", {}).keys())


def get_prompt_for_task(task: str, meme_text: str, image_description: str, few_shot_examples: str, english_dataset: bool = True) -> str:
    config = get_classification_config()
    tasks_config = config.get("tasks", {}) if english_dataset else config.get("polish_tasks", {})

    if task not in tasks_config:
        raise ValueError(
            f"Unknown task: {task}. Available: {list(tasks_config.keys())}"
        )

    template = tasks_config[task].get("user_prompt_template", "")
    return template.format(meme_text=meme_text, image_description=image_description, few_shot_examples=few_shot_examples)


def get_positive_keywords(task: str) -> list:
    keywords = {
        "humour": ["funny", "yes", "true", "hilarious", "amusing", "laugh", "1"],
        "sarcasm": ["sarcastic", "yes", "true", "irony", "ironic", "1"],
        "offensive": ["offensive", "yes", "true", "insulting", "hurtful", "1"],
        "motivational": ["motivational", "yes", "true", "inspiring", "encouraging", "1"],
        "zabawny": ["zabawny", "tak", "niezabawny", "1"],
        "obrazliwy": ["obrazliwy", "tak", "1"],
        "osobisty": ["osobisty", "tak", "1"],
        "zaskakujacy": ["zaskakujacy", "tak", "1"],
    }

    return keywords.get(task, [])


def get_negative_keywords(task: str) -> list:
    keywords = {
        "humour": ["not funny", "no", "false", "not humorous", "0"],
        "sarcasm": ["not sarcastic", "no", "false", "not ironic", "0"],
        "offensive": ["not offensive", "no", "false", "not insulting", "0"],
        "motivational": ["not motivational", "no", "false", "not inspiring", "0"],
        "zabawny": ["nie zabawny", "nie", "niezabawny", "0"],
        "obrazliwy": ["nie obrazliwy", "nie", "nieobrazliwy", "0"],
        "osobisty": ["nie osobisty", "nie", "nieosobisty", "0"],
        "zaskakujacy": ["nie zaskakujacy", "nie", "niezaskakujacy", "0"],
    }

    return keywords.get(task, [])


def get_few_shot_examples_with_images(
    train_df,
    task: str,
    images_base_path: str,
    n_shots: int = 2,
) -> List[Dict]:

    shots_per_class = max(1, n_shots // 2)

    class_0_examples = train_df[train_df[task] == 0].sample(
        n=min(shots_per_class, len(train_df[train_df[task] == 0])),
        random_state=42
    )

    class_1_examples = train_df[train_df[task] == 1].sample(
        n=min(shots_per_class, len(train_df[train_df[task] == 1])),
        random_state=42
    )

    examples = pd.concat([class_0_examples, class_1_examples])

    few_shot_list = []

    for _, row in examples.iterrows():
        image_filename = row.get("image_name") or row.get("image")
        if pd.isna(image_filename):
            logger.warning(f"Skipping example without image: {row.get('text_corrected')}")
            continue

        image_path = os.path.join(images_base_path, str(image_filename))

        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue

        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to encode image {image_path}: {e}")
            continue

        few_shot_list.append({
            "meme_text": row.get("text_corrected", "[no text]"),
            "image_base64": image_base64,
            "image_path": image_path,
            "label": int(row[task]),
        })

    return few_shot_list


def parse_classification_response(response: str, task: str) -> int:
    if response == "error":
        return -1

    response_lower = response.lower().strip()

    for keyword in get_negative_keywords(task):
        if keyword.lower() in response_lower:
            return 0

    for keyword in get_positive_keywords(task):
        if keyword.lower() in response_lower:
            return 1

    logger.warning(f"Could not parse response for task '{task}': '{response}'")
    return -1


def get_few_shot_examples(train_df, task: str, n_shots: int = 2) -> str:
    shots_per_class = max(1, n_shots // 2)
    class_0_examples = train_df[train_df[task] == 0].sample(
        n=min(shots_per_class, len(train_df[train_df[task] == 0])),
        random_state=42
    )
    class_1_examples = train_df[train_df[task] == 1].sample(
        n=min(shots_per_class, len(train_df[train_df[task] == 1])),
        random_state=42
    )
    examples = pd.concat([class_0_examples, class_1_examples])

    few_shot_text = (
        f"Here are {len(examples)} examples of how to classify memes for '{task}':\n\n"
    )
    class_1_examples = train_df[train_df[task] == 1].sample(
        n=min(shots_per_class, len(train_df[train_df[task] == 1])),
        random_state=42
    )
    examples = pd.concat([class_0_examples, class_1_examples])

    few_shot_text = ""

    for idx, (_, row) in enumerate(examples.iterrows(), 1):
        label = row[task]
        meme_text = row.get("text_corrected", "[no text]")
        few_shot_text += f"Example {idx}: \nmeme text: \"{meme_text}\" → label: {label}\n"

    return few_shot_text
