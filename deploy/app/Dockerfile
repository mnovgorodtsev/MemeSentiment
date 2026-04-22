FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY deploy/app/requirements.txt requirements.txt
COPY app ./app
COPY config ./config

ENV PYTHONPATH=/app/app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "app/frontend/app.py"]