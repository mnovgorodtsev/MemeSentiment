# Development & Maintenance

> How to develop and maintain?

## Project structure

```
MemeSentiment/
├── classic_analysis/        # Classical multimodal pipeline
├── docs/                    # Documentation
├── vllm_analysis/           # VLLM benchmark pipeline
├── utils/                   # Dataset download, config helpers
├── config/                  # Hyperparameter grid configs
├── results/                 # Experiment outputs
├── Taskfile.yml             # Task automation
├── Dockerfile
└── docker-compose.yml
```

## Dependencies

All Python dependencies are pinned in `requirements.txt`. After changing them, rebuild the image:

```bash
task build
```

Key libraries: `torch`, `transformers`, `ollama`, `scikit-learn`, `mlflow`.

## Adding a new classical model

1. Create a new module under `classic_analysis/`, e.g. `classic_analysis/clip_pipeline/`
2. Add a config file under `config/`
3. Register `train` and `test` tasks in `Taskfile.yml`

## Adding a new VLLM model

1. Add the model name to the relevant config or pass it via `--model_type` argument in `vllm_analysis/vllm_analysis.py`
2. Pull the model in Ollama: `docker compose exec ollama ollama pull <model>` 
3. You can also use our OpenAI provider to launch and run models
4. You can also use our LammaCpp provider to launch and run models

## Reproducibility

Classical pipeline results are fully reproducible via a fixed random seed. VLLM results may vary slightly between runs due to non-deterministic inference.

## Environment

- All workloads run inside Docker containers — no local Python environment needed
- The `app` service runs the training/evaluation code
- The `ollama` service serves VLLM models over a local HTTP API
- Both services communicate over an internal Docker network
