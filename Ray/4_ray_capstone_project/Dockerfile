FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --default-timeout=300 --retries 10 \
    ray==2.39.0 \
    pandas \
    pyarrow \
    numpy
