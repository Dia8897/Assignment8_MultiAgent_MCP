FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY mcp_server/requirements.txt /app/mcp_server/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /app/mcp_server/requirements.txt

COPY mcp_server/server.py /app/mcp_server/server.py

EXPOSE 8000

CMD ["python", "mcp_server/server.py"]