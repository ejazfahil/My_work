# Convenience image. The verified path in the README is a local venv + pip.
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m pytest -q
CMD ["python", "benchmarks/run_benchmark.py"]
