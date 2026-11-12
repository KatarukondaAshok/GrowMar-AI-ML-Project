# Serves the Phase 2 API on port 8000.  Build: docker build -t aura-phase2 .   Run: docker run -p 8000:8000 aura-phase2
# The chatbot needs the HR system: set HR_SERVICE_URL (default http://127.0.0.1:9000, see mock_services/).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app ./app
COPY models ./models
COPY mock_services ./mock_services
ENV AURA_MODELS_DIR=models AURA_DEMO_TODAY=2026-09-30
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
