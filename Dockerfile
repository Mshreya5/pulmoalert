FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Ensure env package is importable
RUN touch env/__init__.py

# Real-time stdout — critical for [STEP] line streaming during evaluation
ENV PYTHONUNBUFFERED=1

# The evaluator calls `python inference.py` directly.
# The server is for the HF Space interactive UI only.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
