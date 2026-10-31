# model-runtime 이미지 (가짜 백엔드 기본). GPU PC는 Windows에 직접 설치해 쓴다 (README).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 WORKER_BACKEND=fake
RUN useradd --system --uid 10001 app
WORKDIR /app
COPY pyproject.toml ./
COPY model_runtime ./model_runtime
RUN pip install --no-cache-dir .
USER 10001
ENTRYPOINT ["python", "-m", "model_runtime"]
