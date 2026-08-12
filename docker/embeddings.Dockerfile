# BGE embedding + reranking service, weights baked in.
#
# The bake is the whole point: it costs a ~4 GB image and a slow first build,
# and buys an image that runs with no network at all. HF_HUB_OFFLINE=1 is set
# below rather than only in compose, so a regression that reintroduced a
# runtime download fails inside this image too.
#
# Build:  docker compose build embeddings      (~6-9 min cold)
# Rebuild is only needed when a *_MODEL arg changes.

FROM python:3.12-slim

ARG EMBEDDING_MODEL=BAAI/bge-m3
ARG RERANKER_MODEL=BAAI/bge-reranker-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    RERANKER_MODEL=${RERANKER_MODEL}

WORKDIR /app

# curl is only here for the healthcheck; keeping the layer minimal otherwise.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch from the CPU index first so the amd64 build does not pull the
# ~2.5 GB of bundled CUDA libraries it can never use here -- containers on
# macOS have no GPU passthrough, and CI runners have no GPU either.
COPY docker/embeddings-requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "$(grep '^torch==' /tmp/requirements.txt)" \
    && pip install -r /tmp/requirements.txt

# Bake the weights. Instantiating the real classes (rather than calling
# snapshot_download with a hand-written allow_patterns) guarantees we fetch
# exactly the files these classes load at runtime -- an allow_patterns list
# that misses one file produces an image that only fails on first request.
RUN python - <<'PY'
import os
from sentence_transformers import CrossEncoder, SentenceTransformer

emb = os.environ["EMBEDDING_MODEL"]
rer = os.environ["RERANKER_MODEL"]
print(f"baking embedding model: {emb}", flush=True)
m = SentenceTransformer(emb, device="cpu")
print(f"  dim={m.get_sentence_embedding_dimension()}", flush=True)
print(f"baking reranker: {rer}", flush=True)
CrossEncoder(rer, device="cpu", max_length=512)
print("bake complete", flush=True)
PY

# Only now is the download forbidden -- set after the bake so the bake can run.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY src/edgar_eval/embed/server.py /app/server.py

EXPOSE 8081

# Single worker on purpose: each worker would hold its own copy of ~3.4 GB of
# weights, and the bottleneck is CPU-bound inference that threads already use.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1"]
