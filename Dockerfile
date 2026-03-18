FROM python:3.11-slim

# Build dependencies for numpy, scipy, cvxpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8050
EXPOSE ${PORT}

CMD ["gunicorn", "app:server", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:8050"]
