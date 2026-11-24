FROM python:3.12-slim-bookworm AS dependencies

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app \
    && install -d -o app -g app /var/lib/mlflow /var/lib/mlflow/artifacts

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY --from=dependencies /opt/venv /opt/venv
COPY app/ ./app/
COPY fairness/ ./fairness/
COPY healthequity/ ./healthequity/
COPY evals/ ./evals/
COPY docs/methodology.md ./docs/methodology.md
COPY data/samples/diabetes_audit_sample.csv data/samples/diabetes_audit_sample_metadata.json ./data/samples/
COPY config.yaml streamlit_app.py ./

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8501') + '/_stcore/health', timeout=3).close()"]

CMD ["sh", "-c", "exec python -m streamlit run streamlit_app.py --server.port=\"${PORT:-8501}\""]
