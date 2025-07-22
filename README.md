# Generative-AI-for-Synthetic-Document-Generation-and-Analysis
DocGenAI
A web-based application for generating synthetic documents and analyzing uploaded documents using generative AI.
Features

Document Generation: Create reports, articles, or other text using a fine-tuned GPT-2 model.
Document Analysis: Extract text and entities from images using Tesseract and spaCy.
Tech Stack: FastAPI, Hugging Face Transformers, Pytesseract, spaCy, Docker.

Setup

Clone the repository:git clone https://github.com/superuser303/DocGenAI.git
cd DocGenAI


Install dependencies:pip install -r requirements.txt


Install Tesseract OCR:
Ubuntu: sudo apt-get install tesseract-ocr
macOS: brew install tesseract


Download spaCy model:python -m spacy download en_core_web_sm


Run locally with Docker:docker-compose up --build


Access at http://localhost:8000.

Model Training

Document Generation:
Fine-tune GPT-2:python app/train_generator.py


Dataset: Place sample documents in data/reports.txt.


Document Analysis:
Uses pre-trained spaCy (en_core_web_sm) for entity extraction.



Testing

Test generation:python app/test_generator.py


Test analysis:python app/test_analyzer.py



Deployment
Deploy to Heroku:
heroku login
heroku create docgenai-superuser303
heroku container:push web
heroku container:release web
heroku open

Demo
Live Demo (Add Heroku URL after deployment)
Contributing
Fork and submit pull requests to enhance features!
License
MIT License