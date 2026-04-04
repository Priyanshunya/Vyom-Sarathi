# pinned base requirement
FROM ubuntu:22.04

# suppress interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# core runtime
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# build dependency layer first for cache efficiency
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# mount application source
COPY . .

# api ingress
EXPOSE 8000

# boot daemon
CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8000"]
