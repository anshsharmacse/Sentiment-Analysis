from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import wandb

# Initialize the FastAPI app
app = FastAPI()

# Load the fine-tuned model and tokenizer (replace with your saved model path)
model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert")
tokenizer = BertTokenizer.from_pretrained("./fine_tuned_bert")

# Set up wandb (optional for experiment tracking)
wandb.init(project="sentiment-analysis-api", name="api-deployment")

# Define the input format using Pydantic
class Review(BaseModel):
    text: str

# Define the sentiment prediction endpoint
@app.post("/predict/")
def predict_sentiment(review: Review):
    # Tokenize input text
    inputs = tokenizer(review.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted sentiment (choose the max logit)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    # Map class index to sentiment label
    sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}
    predicted_sentiment = sentiment[predicted_class]
    
    # Log the prediction to wandb (optional)
    wandb.log({"review": review.text, "predicted_sentiment": predicted_sentiment})
    
    return {"review": review.text, "sentiment": predicted_sentiment}
