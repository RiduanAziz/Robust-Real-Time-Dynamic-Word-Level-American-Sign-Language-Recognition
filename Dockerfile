FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests

RUN pip install --upgrade pip && pip install -e .

CMD ["python", "scripts/train.py", "--config", "configs/base.yaml"]
