# Meal-photo nutrition API, CPU only.
#   docker build -t food-plate-api .
#   docker run -p 8000:8000 food-plate-api            # then open http://localhost:8000/docs
FROM python:3.10-slim

# shared libraries OpenCV needs (OpenCV comes with ultralytics)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0 \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY plate/ plate/
COPY models/yolo/detector.pt models/yolo/detector.pt
COPY models/plate_classifier/ models/plate_classifier/
COPY data/nutrition/dish_nutrition.csv data/nutrition/dish_nutrition.csv

# every model file is in the image: no downloads, no Ultralytics telemetry or run-time pip installs
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 YOLO_OFFLINE=1 YOLO_AUTOINSTALL=False YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "plate.api:app", "--host", "0.0.0.0", "--port", "8000"]
