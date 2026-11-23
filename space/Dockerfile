# The BrickMatch API on a Hugging Face Space (free CPU hardware).
# .github/workflows/deploy-space.yml copies the code next to this file and uploads it to the Space.
FROM python:3.11-slim

# OpenCV (used by ultralytics) needs these system libraries.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the app as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r server/requirements.txt

COPY --chown=user . /app
USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# Build pictures and the catalog the site reads.
RUN python -m server.site_assets

EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
