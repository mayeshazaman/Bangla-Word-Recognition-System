# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Bangla OCR Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -t bangla-ocr .
#
# Run (after training on the host):
#   docker run -p 8501:8501 \
#     -v "$(pwd)/models:/app/models:ro" \
#     -v "$(pwd)/labels.json:/app/labels.json:ro" \
#     -v "$(pwd)/artifacts:/app/artifacts:ro" \
#     bangla-ocr
#
# Demo mode (no artefacts — random predictions):
#   docker run -p 8501:8501 bangla-ocr
#
# Then open http://localhost:8501 in your browser.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .

# Swap GUI OpenCV for the headless build (no display needed inside Docker)
RUN sed -i 's/opencv-python==/opencv-python-headless==/' requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY app.py .

# ── Directory layout matching the project structure ───────────────────────────
# models/      → mount at runtime: -v "$(pwd)/models:/app/models:ro"
# artifacts/   → mount at runtime: -v "$(pwd)/artifacts:/app/artifacts:ro"
# labels.json  → mount at runtime: -v "$(pwd)/labels.json:/app/labels.json:ro"
#
# Create empty placeholders so the app starts in demo mode even without mounts.
RUN mkdir -p models artifacts/mlflow && \
    touch labels.json

# ── Streamlit configuration ───────────────────────────────────────────────────
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD wget -qO- http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
