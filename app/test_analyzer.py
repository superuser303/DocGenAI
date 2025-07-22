import pytesseract
from PIL import Image
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Test image analysis
test_image = "test_data/sample_document.jpg"
text = pytesseract.image_to_string(Image.open(test_image))
doc = nlp(text)

# Extract entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(f"Extracted Text: {text}")
print(f"Entities: {entities}")