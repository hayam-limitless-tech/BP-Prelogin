FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY adapter.py .

# Railway provides PORT at runtime
ENV PORT=8000

CMD ["sh", "-c", "uvicorn adapter:app --host 0.0.0.0 --port ${PORT}"]
