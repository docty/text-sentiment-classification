from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize FastAPI app
app = FastAPI()

# Define request body schema
class TextInput(BaseModel):
    text: str

# Define sentiment analysis function
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    sentiment = "positive" if predicted_class == 1 else "negative"
    return sentiment, probs.tolist()

# Define API endpoint
@app.post("/analyze-sentiment")
async def analyze_sentiment_endpoint(text_input: TextInput):
    try:
        sentiment, probs = analyze_sentiment(text_input.text)
        return {
            "text": text_input.text,
            "sentiment": sentiment,
            "probabilities": probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}