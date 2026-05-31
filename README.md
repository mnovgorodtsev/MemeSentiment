# MemSen: Classical Multimodal Methods vs Vision-Language Large Models

## Project Overview

This repository contains the implementation and experimental framework for a **Master's thesis** focused on **sentiment analysis of internet memes**.  
The main objective of the project is to **compare classical multimodal sentiment analysis approaches** with **modern Vision-Language Large Models** and analyze their effectiveness, limitations, and behavior across different types of memes.

---

## Requirements

- [Docker](https://www.docker.com/products/docker-desktop) + [Docker Compose](https://docs.docker.com/compose/)
- [Task](https://taskfile.dev/installation/) (task runner)

---

## Quick Start

The demo runs a representative subset of experiments - fast enough to verify the setup without a full training run. The first build should take around ***10 minutes***. 

> ⚠️ We recommend using a machine equipped with a GPU for optimal performance. However, we have configured the demo to run reasonably fast even on CPU-only systems.

### 1. Clone the repository

```bash
git clone https://github.com/mnovgorodtsev/MemeSentiment
cd MemeSentiment
```

### 2. Run VLLM demo

```bash
task vllm        # english dataset, 5 samples
task vllm-pl     # polish dataset, 5 samples
```

This downloads and extracts both the English (Memotion 7k) and Polish meme datasets, download small gemma3:latest model on Ollama and run sample experiment. You can increase number of samples in Taskfile.yml by changing the variable SAMPLES.

After the process finishes, a results file will be available in results/vllms/demo. The results should closely resemble those reported in the GitHub repository. However, slight variations are expected, as VLLM inference cannot be fully controlled through deterministic random seeds.

### 3. Run Classical pipeline demo

Train demo models (1 epoch, lightweight config):

```bash
task classic-demo-train
```

You should see logs after running the process, and at the end you should find a results file in results/demo/, which should match the one posted on GitHub due to the fixed seed.

Test demo models:

```bash
task classic-demo-test
```

You should see logs after running the process, and at the end you should find a results file in results/demo/test, which should match the one posted on GitHub due to the fixed seed.

---

## Full pipeline

Each model has its own task, for example, if you would like to run the whole pipeline for BERT:

> ⚠️ Full training takes several hours even with GPU.

```bash
task bert-train
task bert-test   
```

---

## Project Structure

```
MemeSentiment/
├── classic_analysis/          # Classic benchmark pipeline
├── vllm_analysis/             # VLLM benchmark pipeline
├── utils/                     # Dataset download, config utilities
├── data/                      # Datasets (not tracked in git)
├── models/                    # Saved model weights (not tracked in git)
├── results/                   # Experiment results
├── config/                    # Hyperparameter configs
├── Taskfile.yml
├── Dockerfile
└── docker-compose.yml
```

---

## Available Tasks

| Task | Description |
|---|---|
| `task dataset` | Download both datasets (English + Polish) |
| `task build` | Build Docker image |
| `task setup` | Start Ollama, pull model and download datasets |
| `task classic-demo-train` | Train one model per pipeline, 1 epoch, single config |
| `task classic-demo-test` | Test models trained in demo |
| `task vllm` | VLLM benchmark, English dataset (5 samples) |
| `task vllm-pl` | VLLM benchmark, Polish dataset (5 samples) |
| `task bert-train` | Full BERT training (grid search) |
| `task bert-test` | BERT evaluation |
| `task resnet-train` | Full ResNet training (grid search) |
| `task resnet-test` | ResNet evaluation |
| `task early-train` | Early fusion training (grid search) |
| `task early-test` | Early fusion evaluation |
| `task late-train` | Late fusion training (requires trained BERT + ResNet) |
| `task late-test` | Late fusion evaluation |

---

## Research Objectives

- Build a **classical multimodal sentiment analysis pipeline** for memes using separate vision and language models
- Compare classical methods with **state-of-the-art Vision-Language Large Models**
- Perform a **fine-grained behavioral analysis of VLLMs**
- Evaluate model robustness to relatability, offensiveness, humor, and unexpectedness content

---

## Part I - Classical Multimodal Sentiment Analysis

This part focuses on building a traditional multimodal representation and fusion pipeline.

### Methodology

- **Text Encoder** — BERT-based model for extracting textual representations from meme captions
- **Image Encoder** — ResNet-based convolutional neural network for visual feature extraction
- **Multimodal Fusion Strategies**
  - Early fusion
  - Late fusion

---

## Part II - Vision-Language Large Models

This part is dedicated to an in-depth study of modern Vision-Language Large Models and their behavior in meme sentiment analysis tasks.

### [Memotion Dataset](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k)

- Evaluation of multiple Vision-Language Large Models
- Prompt engineering and output interpretation strategies
- Analysis of model sensitivity to meme characteristics: humor, relatability, unexpectedness, offensive content
- Comparison between zero-shot and few-shot inference

### [Polish Dataset](https://www.kaggle.com/datasets/kaszkai/polish-meme-dataset)

To evaluate model performance in a non-English context, a custom **Polish meme dataset** was collected and annotated.

- 100 samples labeled by external annotators
- Each meme rated across four dimensions: **funny**, **offensive**, **personal**, and **surprising**
- Enables cross-lingual comparison of VLLM behavior on culturally specific content

---

## Authors

**Katarzyna Michalska** · **Matwej Novgorodtsev**  
Master's Thesis — Adam Mickiewicz University in Poznań
