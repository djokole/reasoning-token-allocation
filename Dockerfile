FROM python:3.11-slim

RUN apt update && apt upgrade -y && apt install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY token_env ./token_env

EXPOSE 8080

CMD ["python", "server.py"]
