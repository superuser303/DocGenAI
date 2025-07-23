FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y tesseract-ocr
COPY . .
RUN python -m spacy download en_core_web_sm
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]