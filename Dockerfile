# ==============================================================================
# Vyom-Sarathi ACM - Production Container
# Architecture: Ubuntu 22.04 Runtime (NSH 2026 Compliant)
# ==============================================================================

# Strictly pinned to Ubuntu 22.04 as mandated by NSH 2026 Section 8.
# Ensures deterministic builds and prevents dependency drift during auto-grading.
FROM ubuntu:22.04

# Suppress interactive prompts (e.g., tzdata) to prevent build pipeline hangs.
# Force Python stdout/stderr to be unbuffered for real-time grader logging.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Initialize core system dependencies.
# Apt caches are explicitly cleared within the same layer to minimize the final image footprint.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Establish the primary operational directory for the payload
WORKDIR /app

# Isolate requirement installation to a dedicated Docker layer.
# This maximizes build cache efficiency during iterative astrodynamics engine updates.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Mount the complete application source (Astrodynamics Engine + API + Visualizer)
COPY . .

# Expose the designated ingress port for the hackathon evaluation harness
EXPOSE 8000

# Boot the high-performance ASGI daemon.
# Explicitly bound to 0.0.0.0 to guarantee external network visibility for the grader scripts.
CMD ["python3", "-m", "uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8000"]
