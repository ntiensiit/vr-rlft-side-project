FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --upgrade pip && pip install uv && \
    uv sync --frozen --all-extras

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY deploy/robot.xml ./deploy/robot.xml

ENV PYTHONPATH=/app/src

CMD ["python", "scripts/run_artifacts.py"]