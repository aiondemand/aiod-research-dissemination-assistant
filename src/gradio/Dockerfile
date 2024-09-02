FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run application
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0"]
CMD ["--port", "7860", "--log-level", "debug"]
