FROM python:3.11-slim
WORKDIR /srv
# libgomp1 is required by LightGBM (OpenMP); slim images don't ship it
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/srv/app
EXPOSE 8000
# shell form so $PORT (injected by Render/hosts) is expanded
CMD uvicorn main:app --app-dir app --host 0.0.0.0 --port ${PORT:-8000}
