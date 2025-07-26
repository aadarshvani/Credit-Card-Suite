FROM python:3.10-slim

WORKDIR /app
COPY frontend/ ./frontend/
COPY artifacts/ ./artifacts/
COPY models/ ./models/
WORKDIR /app/frontend

RUN pip install --upgrade pip
RUN pip install streamlit requests

CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"] 