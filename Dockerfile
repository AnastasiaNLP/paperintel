FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl passwd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 paperintel

COPY --chown=paperintel:paperintel . .

USER paperintel

EXPOSE 8000

CMD ["uvicorn", "api.rest.main:app", "--host", "0.0.0.0", "--port", "8000"]
