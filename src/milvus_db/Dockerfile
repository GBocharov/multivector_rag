FROM python:3.10-slim

WORKDIR /opt/app-root/src/

RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pdf2image python-dateutil

ENV POPPLER_PATH="/usr/bin"

COPY src/milvus_db/requirements.txt /opt/app-root/src/milvus_db/

RUN pip install -r /opt/app-root/src/milvus_db/requirements.txt

ENV PYTHONPATH /opt/app-root/src
ENV PYTHONUNBUFFERED=1