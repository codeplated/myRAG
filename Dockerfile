FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY config ./config
COPY eval ./eval

# Run as a normal user, not root. Uploaded PDFs and logs go to these two folders.
RUN useradd --create-home appuser \
    && mkdir -p /app/data/books /app/logs \
    && chown -R appuser:appuser /app/data /app/logs /app/eval
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
