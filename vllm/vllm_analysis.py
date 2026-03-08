import gradio as gr
import ollama
import configparser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.ini"

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

OLLAMA_HOST = config["ollama"]["host"]
MODEL_NAME = config["ollama"]["model"]

SERVER_HOST = config["server"]["host"]
SERVER_PORT = int(config["server"]["port"])

client = ollama.Client(host=OLLAMA_HOST)

prompts = [
    "Classify if this meme is funny or not funny. Return only class: funny or not funny",
    "Classify if this meme is general, twisted or not sarcastic. Return only class: general, twisted or not sarcastic",
    "Classify if this meme is offensive or not offensive. Return only class: offensive or not offensive",
    "Classify if this meme is motivational or not motivational. Return only class: motivational or not motivational"
]

def classify_meme(image):
    results = []

    for prompt in prompts:
        response = client.chat(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image]
            }]
        )

        results.append(response["message"]["content"])

    return {
        "funny": results[0],
        "sarcasm": results[1],
        "offensive": results[2],
        "motivational": results[3],
    }


with gr.Blocks() as demo:
    gr.Markdown("# Meme classifier")

    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload meme")

    output = gr.JSON(label="Classification")

    btn = gr.Button("Classify")

    btn.click(
        fn=classify_meme,
        inputs=image_input,
        outputs=output
    )

demo.launch(
    server_name=SERVER_HOST,
    server_port=SERVER_PORT
)