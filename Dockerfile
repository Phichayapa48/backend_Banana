FROM python:3.11-slim

# Linux deps สำหรับ OpenCV (headless)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ใช้ PORT จาก Render
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
