# Base environment pinned to NSH 2026 specifications
FROM ubuntu:22.04

# Suppress prompts for automated grading
ENV DEBIAN_FRONTEND=noninteractive

# Python optimization: disable bytecode and enable unbuffered logging for telemetry
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime and build-essential for Numba JIT-compilation support
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependency layer
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy ACM source (core/ and dashboard/)
COPY . .

# Expose port 8000 for simulation engine ingress
EXPOSE 8000

# Bind to 0.0.0.0 to allow external grader communication
CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8000"]
