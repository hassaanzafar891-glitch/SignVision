FROM python:3.10-slim-bookworm
WORKDIR /app
# MediaPipe/OpenCV and the retained legacy speech support need native libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libportaudio2 libasound2 libsndfile1 espeak-ng libespeak1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=1000 --retries=10 -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
