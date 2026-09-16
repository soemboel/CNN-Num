FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless still links against these even without GUI support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects the container to listen on this port
ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
