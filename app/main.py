from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pytesseract
from PIL import Image
import io
import spacy

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load fine-tuned model
generator = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
nlp = spacy.load("en_core_web_sm")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate_document(prompt: str = Form(...)):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = generator.generate(**inputs, max_length=200)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        text = pytesseract.image_to_string(image)
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        structured_data = {
            "text": text,
            "entities": entities,
            "tables": []  # Add donut or layoutparser later
        }
        return {"extracted_data": structured_data}
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}