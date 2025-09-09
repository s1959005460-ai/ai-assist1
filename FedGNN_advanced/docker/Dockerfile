# docker/Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y build-essential \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

EXPOSE 50051 8000

CMD ["python", "grpc_server.py"]
