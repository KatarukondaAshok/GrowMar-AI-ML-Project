# Runs the Phase 3 tests in a clean container.  Build: docker build -t aura-phase3 .   Run: docker run aura-phase3
# (not built or run while preparing this phase)
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY src ./src
COPY models ./models
COPY tests ./tests
COPY pytest.ini .
CMD ["pytest", "-q"]
