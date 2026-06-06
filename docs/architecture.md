# Architecture

# Main components

|Component|Role|Technology|
|-|-|-|
|`classic\_analysis/bert\_pipeline`|Text encoder — fine-tuned on meme captions|BERT, HuggingFace Transformers|
|`classic\_analysis/resnet\_pipeline`|Image encoder — fine-tuned on meme images|ResNet, PyTorch / torchvision|
|`classic\_analysis/resnet\_bert\_fusion`|Combines text + image representations|PyTorch|
|`vllm\_analysis`|End-to-end multimodal inference via VLLM|Ollama, OpenAI-compatible API|
|`utils`|Dataset download and config loading|Kaggle API, Python|
|`config/`|Hyperparameter grids|YAML|
|`results/`|Metrics, logs, model artifacts|MLflow, JSON/CSV|




