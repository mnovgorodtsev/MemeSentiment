import logging
from typing import Dict
from utils.read_config import Config

logger = logging.getLogger(__name__)


def get_classification_config() -> Dict:
    config = Config()
    return config.classification_config


def get_tasks() -> list:
    config = get_classification_config()
    return list(config.get("tasks", {}).keys())


def get_prompt_for_task(task: str, meme_text: str, image_description: str, few_shot_examples: str) -> str:
    config = get_classification_config()
    tasks_config = config.get("tasks", {})

    if task not in tasks_config:
        raise ValueError(
            f"Unknown task: {task}. Available: {list(tasks_config.keys())}"
        )

    template = tasks_config[task].get("user_prompt_template", "")
    return template.format(meme_text=meme_text, image_description=image_description, few_shot_examples=few_shot_examples)


def get_positive_class(task: str) -> str:
    config = get_classification_config()
    tasks_config = config.get("tasks", {})
    return tasks_config.get(task, {}).get("positive_class", "")


def get_categories(task: str) -> list:
    config = get_classification_config()
    tasks_config = config.get("tasks", {})
    return tasks_config.get(task, {}).get("categories", [])


def get_positive_keywords(task: str) -> list:
    keywords = {
        "humour": ["funny", "yes", "true", "hilarious", "amusing", "laugh", "1"],
        "sarcasm": ["sarcastic", "yes", "true", "irony", "ironic", "1"],
        "offensive": ["offensive", "yes", "true", "insulting", "hurtful", "1"],
        "motivational": ["motivational", "yes", "true", "inspiring", "encouraging", "1"],
    }

    return keywords.get(task, [])


def get_negative_keywords(task: str) -> list:
    keywords = {
        "humour": ["not funny", "no", "false", "not humorous", "0"],
        "sarcasm": ["not sarcastic", "no", "false", "not ironic", "0"],
        "offensive": ["not offensive", "no", "false", "not insulting", "0"],
        "motivational": ["not motivational", "no", "false", "not inspiring", "0"],
    }

    return keywords.get(task, [])


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
    examples = train_df.sample(n=min(n_shots, len(train_df)), random_state=42)
    
    few_shot_text = "Here are some examples:\n\n"

    for idx, (_, row) in enumerate(examples.iterrows(), 1):
        label = row[task]
        meme_text = row.get("text_corrected", "[no text]")
        few_shot_text += f"Example {idx}: \nmeme text:\"{meme_text}\" → label: {label}\n"

    return few_shot_text
