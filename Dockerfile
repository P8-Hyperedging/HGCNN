FROM python:3.14-slim

WORKDIR /app

# Copy and install requirements in a separate layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

CMD ["python", "-u", "src/flaskeladen.py"]