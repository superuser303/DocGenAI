FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository universe
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils imagemagick
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m spacy download en_core_web_sm
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]