from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Example synthetic dataset
data = {
    "ReviewText": [
        "The event was amazing and well-organized!",
        "The amenities were not up to the mark.",
        "Had a decent experience overall.",
        "The campus event was a complete disaster.",
        "I loved every moment of the event!"
    ],
    "SentimentLabel": [2, 0, 1, 0, 2]  # 0: Negative, 1: Neutral, 2: Positive
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split into train, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["ReviewText"], df["SentimentLabel"], test_size=0.2, random_state=42
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=42
)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
