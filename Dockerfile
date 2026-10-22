# syntax=docker/dockerfile:1.7
# Two targets:
#   docker build -t bioretrieval-mcp .                 # slim: EMDB / UniProt / GEO tools
#   docker build --target ml -t bioretrieval-mcp:ml .  # + ESM-2 embeddings and map retrieval (CPU torch)

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    BIORETRIEVAL_CACHE_DIR=/data/cache
RUN useradd --create-home --uid 1000 app && mkdir -p /data/cache && chown -R app /data
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .
USER app
VOLUME ["/data"]
EXPOSE 8000
# Streamable HTTP by default; override with `--transport stdio` for local MCP clients.
ENTRYPOINT ["bioretrieval-mcp"]
CMD ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ml
USER root
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install ".[ml]"
# ESM-2 weights (~2.5 GB) download on first use into TORCH_HOME; mount /data to keep them.
ENV TORCH_HOME=/data/torch
USER app
