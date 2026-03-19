FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8050
EXPOSE $PORT

CMD ["sh", "-c", "gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120"]
