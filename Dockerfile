# CareMate perception service: OpenCV 5 ALIKED + USAC target verification over HTTP.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

# ALIKED ONNX weights (third-party export), pinned by checksum.
ARG ALIKED_URL=https://huggingface.co/bukuroo/ALIKED-LightGlue-ONNX/resolve/main/aliked-n16rot-top1k-640.onnx
ARG ALIKED_SHA256=b46868de0794ea0f2d878985d99dc17d3639dfab85d852994254a4cdf0cdc0ea
RUN mkdir -p models/vision \
    && curl -fsSL -o models/vision/aliked-n16rot-top1k-640.onnx "$ALIKED_URL" \
    && echo "$ALIKED_SHA256  models/vision/aliked-n16rot-top1k-640.onnx" | sha256sum -c -

COPY caremate/__init__.py caremate/perception_v0_7.py caremate/workspace_v0_7.py caremate/enrol_v0_7.py caremate/
COPY caremate/service caremate/service
COPY references/v0_7 references/v0_7

RUN useradd --system --uid 10001 caremate && mkdir -p /data/references && chown caremate /data/references
USER caremate
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CAREMATE_REF_STORE=/data/references
VOLUME /data/references
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fs http://localhost:8000/v1/health || exit 1
CMD ["uvicorn", "caremate.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
