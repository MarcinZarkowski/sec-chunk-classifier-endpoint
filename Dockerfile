FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install uv
RUN pip install uv

# Install dependencies using uv sync (system-wide so it uses the system Python environment)
COPY pyproject.toml uv.lock ./
RUN uv sync --system

# Download and bake the model into the image
COPY download_model.py .
RUN python download_model.py

# Copy the handler
COPY handler.py .

# Start the handler
CMD ["python", "handler.py"]
