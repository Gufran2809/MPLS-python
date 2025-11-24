FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements_distributed.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements_distributed.txt

# Copy source code
COPY src/ ./src/
COPY proto/ ./proto/
COPY scripts/ ./scripts/

# Generate gRPC stubs
RUN python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./src/distributed \
    --grpc_python_out=./src/distributed \
    ./proto/mpls_service.proto

# Create data and logs directories
RUN mkdir -p /app/data /app/logs /app/configs

# Expose port range for workers
EXPOSE 50051-50060

# Default command (will be overridden in docker-compose)
CMD ["python", "-m", "src.distributed.worker"]
