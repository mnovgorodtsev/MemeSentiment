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


def get_prompt_for_task(task: str, meme_text: str) -> str:
    config = get_classification_config()
    tasks_config = config.get("tasks", {})

    if task not in tasks_config:
        raise ValueError(
            f"Unknown task: {task}. Available: {list(tasks_config.keys())}"
        )

    template = tasks_config[task].get("user_prompt_template", "")
    return template.format(meme_text=meme_text)


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
        "humour": ["funny", "yes", "true", "hilarious", "amusing", "laugh"],
        "sarcasm": ["sarcastic", "yes", "true", "irony", "ironic"],
        "offensive": ["offensive", "yes", "true", "insulting", "hurtful"],
        "motivational": ["motivational", "yes", "true", "inspiring", "encouraging"],
    }

    return keywords.get(task, [])


def get_negative_keywords(task: str) -> list:
    keywords = {
        "humour": ["not funny", "no", "false", "not humorous"],
        "sarcasm": ["not sarcastic", "no", "false", "not ironic"],
        "offensive": ["not offensive", "no", "false", "not insulting"],
        "motivational": ["not motivational", "no", "false", "not inspiring"],
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
    examples = train_df.sample(n=min(n_shots, len(train_df)))
    few_shot_text = (
        f"Here are {len(examples)} examples of how to classify memes for '{task}':\n\n"
    )

    for idx, (_, row) in enumerate(examples.iterrows(), 1):
        label = "yes" if row[task] == 1 else "no"
        meme_text = row.get("text_corrected", "no text")
        few_shot_text += f"{idx}. Text: '{meme_text}' → Answer: {label}\n"

    few_shot_text += "\nNow classify the current image following the same pattern."
    return few_shot_text
