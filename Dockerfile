FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install uv the official Docker way (standalone binary)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies into a virtual environment
COPY pyproject.toml uv.lock ./
RUN uv sync

# Download and bake the model into the image
COPY download_model.py .
RUN python download_model.py

# Copy the handler
COPY handler.py .

# Start the handler
CMD ["uv", "run", "handler.py"]
