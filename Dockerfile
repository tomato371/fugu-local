FROM python:3.12-slim

WORKDIR /app

# Optional dependencies (web UI + multi-format file I/O).
# The core CLI needs none of these — see requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ollama endpoint. Defaults to the host's Ollama (Docker Desktop);
# docker-compose overrides this to reach the bundled `ollama` service.
ENV OLLAMA_URL=http://host.docker.internal:11434

# Gradio web UI
EXPOSE 7860

CMD ["python", "fugu_web.py"]
