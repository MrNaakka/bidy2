FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "mcp>=1.20,<2" pillow
COPY server.py /srv/server.py
CMD ["python", "/srv/server.py"]
