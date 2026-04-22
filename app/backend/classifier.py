import ollama
from utils.read_config import load_config

cfg = load_config()

client = ollama.Client(host=cfg["ollama_host"])
MODEL_NAME = cfg["model_name"]

PROMPTS = [
    "Classify if this meme is funny or not funny. Return only class: funny or not funny",
    "Classify if this meme is general, twisted or not sarcastic. Return only class: general, twisted or not sarcastic",
    "Classify if this meme is offensive or not offensive. Return only class: offensive or not offensive",
    "Classify if this meme is motivational or not motivational. Return only class: motivational or not motivational",
]


def classify_meme(image_path: str):
    results = []

    for prompt in PROMPTS:
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
        )

        results.append(response["message"]["content"].strip())

    return {
        "funny": results[0],
        "sarcasm": results[1],
        "offensive": results[2],
        "motivational": results[3],
    }
