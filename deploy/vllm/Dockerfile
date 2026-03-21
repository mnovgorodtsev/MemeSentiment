FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY deploy/vllm/requirements.txt requirements.txt
COPY vllm ./vllm
COPY config ./config

ENV PYTHONPATH=/app/vllm

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "vllm/frontend/app.py"]