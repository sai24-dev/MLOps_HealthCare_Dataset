FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY baseline-model.py .

CMD ["python", "baseline-model.py"]
