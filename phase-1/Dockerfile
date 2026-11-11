FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt jupyterlab

COPY . .

EXPOSE 8888

# Default: Jupyter Lab on http://localhost:8888. Override the command to run scripts instead, e.g.
#   docker run --rm -v "$PWD:/app" aura-attrition pytest -q
#   docker run --rm -v "$PWD:/app" aura-attrition python -m src.features.cli run --output-dir artifacts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token="]
