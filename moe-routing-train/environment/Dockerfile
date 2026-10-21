FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY train_moe.py /app/train_moe.py
COPY moe_runtime/ /app/moe_runtime/
COPY data/ /app/data/
COPY output.json /app/output.json
RUN chmod 755 /app/train_moe.py
