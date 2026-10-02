FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CURE_LANGUAGE_MODE=local \
    CURE_REAL_CITY=1 \
    CURE_SERVE_WEB=0 \
    ARRIVAL_CORE_DIR=/data/core_store

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --no-deps --only-binary=:all: -r requirements.txt
COPY api_gateway ./api_gateway
COPY arrival ./arrival
COPY addon ./addon
COPY tools ./tools
COPY demo_app.py run.sh ./
RUN chmod +x /app/run.sh && mkdir -p /data/core_store

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))['ready']"
CMD ["./run.sh"]
