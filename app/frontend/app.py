import gradio as gr
from backend.classifier import classify_meme
from backend.config import load_config

cfg = load_config()

with gr.Blocks() as demo:
    gr.Markdown("# Meme classifier")

    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload meme")

    output = gr.JSON(label="Classification")

    btn = gr.Button("Classify")

    btn.click(fn=classify_meme, inputs=image_input, outputs=output)

demo.launch(server_name=cfg["server_host"], server_port=cfg["server_port"])
