FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "src/run_benchmark.py", "--full"]

