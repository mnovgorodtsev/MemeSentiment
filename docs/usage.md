# User Guide

## Requirements

* [Docker](https://www.docker.com/) >= 29.2.1 + Docker Compose >= 5.0.2
* [Task](https://taskfile.dev/) >= 3.51.1

## Setup

```bash
git clone https://github.com/mnovgorodtsev/MemeSentiment
cd MemeSentiment
```

## Running the demo

```bash
task vllm              # VLLM benchmark, English dataset (5 samples)
task vllm-pl           # VLLM benchmark, Polish dataset (5 samples)
task classic-demo-train  # Train one model per pipeline, 1 epoch
task classic-demo-test   # Test demo models
```

> GPU is recommended but not required — the demo runs on CPU in reasonable time.

## Full training

```bash
task bert-train        # Full BERT training (grid search)
task bert-test
task resnet-train      # Full ResNet training (grid search)
task resnet-test
task early-train       # Early fusion
task early-test
task late-train        # Late fusion (requires trained BERT + ResNet)
task late-test
```

## All available tasks

|Command|Description|
|-|-|
|`task dataset`|Download both datasets (EN + PL)|
|`task build`|Build Docker image|
|`task setup`|Start Ollama, pull model, download datasets|
|`task vllm`|VLLM benchmark, English dataset|
|`task vllm-pl`|VLLM benchmark, Polish dataset|
|`task classic-demo-train`|Demo training|
|`task classic-demo-test`|Demo testing|
|`task bert-train` / `bert-test`|BERT pipeline|
|`task resnet-train` / `resnet-test`|ResNet pipeline|
|`task early-train` / `early-test`|Early fusion pipeline|
|`task late-train` / `late-test`|Late fusion pipeline|

To change the number of VLLM samples, edit `SAMPLES` in `Taskfile.yml` (default: `5`).

