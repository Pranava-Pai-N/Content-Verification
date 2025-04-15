import os
import zipfile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class TextInput(BaseModel):
    text: str

MODEL_ZIP_PATH = "Model_Path.zip"
MODEL_DIR = "Model_Path"

if not os.path.exists(MODEL_DIR):  
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
except Exception as e:
    print(f"Error loading model: {e}")


@app.get("/")  
def home():
    return {"message": "FastAPI is running!"}


@app.post("/predict")
def predict(input_text: TextInput):
    try:
        inputs = tokenizer(input_text.text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        label = "Ayurveda" if pred == 1 else "Non-Ayurveda"
        return {"prediction": label}
    except Exception as e:
        return {"error": str(e)}
