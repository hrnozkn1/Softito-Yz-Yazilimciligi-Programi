FROM python:3.11-slim

RUN pip install --no-cache-dir docker

WORKDIR /app
COPY izleyici.py .

CMD ["python", "-u", "izleyici.py"]
