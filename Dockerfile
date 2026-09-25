FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nawah_pipeline.py server.py ./
# Put knowledge_2.json, quality_issues.json, manifest.json in ./data
# before building, or mount them as a volume at runtime.
COPY data/ ./data/

ENV NAWAH_DATA_DIR=/app/data
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
