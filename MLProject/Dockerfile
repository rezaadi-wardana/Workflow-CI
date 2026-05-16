FROM python:3.10-slim

WORKDIR /app

COPY modelling.py .
COPY loan_default_preprocessed.csv .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "modelling.py"]
