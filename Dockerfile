# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files into the container
COPY . .

# Ensure the qdrant directory exists and is writable
RUN mkdir -p /app/qdrant && chmod -R 777 /app/qdrant

# Expose the application port
EXPOSE 7860

# Start the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
