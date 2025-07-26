
FROM python:3.10-slim

WORKDIR /app
COPY backend/ ./backend/
COPY artifacts/ ./artifacts/
COPY models/ ./models/
WORKDIR /app/backend

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
